"""Run local Phase 3 SQLite ingestion, queries, reconciliation and index review."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from .cleaning import DEFAULT_CLEAN_PATH, load_cleaned_dataset
from .data import DEFAULT_RAW_PATH, PROJECT_ROOT
from .database import DEFAULT_DATABASE_PATH, connect_database, load_snapshot
from .phase2 import _write_json
from .sql_analytics import (
    QUERY_QUESTIONS, execute_all_queries, execute_query, reconcile_sql_pandas,
    validate_snapshot_against_frame,
)
from .sql_performance import benchmark_index
from .sql_reporting import reconciliation_markdown, sql_business_report


def run_phase3(
    cleaned_path: str | Path = DEFAULT_CLEAN_PATH,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    reports_dir: str | Path = PROJECT_ROOT / "reports",
    *, benchmark_repeats: int = 100,
) -> dict:
    """Read the existing Phase 2 CSV, never rewrite it, and verify SQL outputs."""
    cleaned_path, database_path, reports_dir = map(Path, (cleaned_path, database_path, reports_dir))
    if not cleaned_path.is_file():
        raise FileNotFoundError(f"Phase 2 CSV missing: {cleaned_path.resolve()}. Run python -m churn_analytics.phase2 first; Phase 3 does not clean a separate copy.")
    if database_path.resolve() == cleaned_path.resolve():
        raise ValueError("Database and processed CSV paths must differ")
    if any(reports_dir.resolve().is_relative_to(PROJECT_ROOT / "data" / name) for name in ("raw", "processed")):
        raise ValueError("Reports must not be written into raw/processed data directories")
    if type(benchmark_repeats) is not int or not 1 <= benchmark_repeats <= 10000:
        raise ValueError("benchmark_repeats must be between 1 and 10000")
    protected = {str(cleaned_path.resolve()): hashlib.sha256(cleaned_path.read_bytes()).hexdigest()}
    if DEFAULT_RAW_PATH.is_file():
        protected[str(DEFAULT_RAW_PATH.resolve())] = hashlib.sha256(DEFAULT_RAW_PATH.read_bytes()).hexdigest()
    frame = load_cleaned_dataset(cleaned_path)
    ingestion = load_snapshot(frame, database_path)
    output = reports_dir / "phase3"
    query_output = output / "query_results"
    query_output.mkdir(parents=True, exist_ok=True)
    with closing(connect_database(database_path, readonly=True)) as conn:
        validate_snapshot_against_frame(conn, frame)
        results = execute_all_queries(conn)
        for name, table in results.items():
            table.to_csv(query_output / f"{name}.csv", index=False)
        comparison = reconcile_sql_pandas(conn, frame)
        comparison.to_csv(output / "sql_pandas_reconciliation.csv", index=False)
        _write_json(output / "sql_pandas_reconciliation.json", comparison.to_dict("records"))
        (output / "sql_pandas_reconciliation.md").write_text(reconciliation_markdown(comparison), encoding="utf-8")
        if not comparison.passed.all():
            raise ValueError("SQL/Pandas reconciliation failed; inspect reports/phase3/sql_pandas_reconciliation.csv")
        performance = benchmark_index(conn, repeats=benchmark_repeats)
    _write_json(output / "query_performance.json", performance)
    (reports_dir / "PHASE3_SQL_BUSINESS_INSIGHTS.md").write_text(
        sql_business_report(results, ingestion, comparison, performance), encoding="utf-8",
    )
    for path, fingerprint in protected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != fingerprint:
            raise RuntimeError(f"Protected source artifact changed: {path}")
    manifest = {
        "ingestion": ingestion, "source_sha256": protected,
        "sqlite_version": sqlite3.sqlite_version, "queries_executed": len(results),
        "query_row_counts": {name: len(table) for name, table in results.items()},
        "sql_kpis": results["01_customer_kpis"].to_dict("records")[0],
        "reconciliation_checks": len(comparison), "reconciliation_passed": int(comparison.passed.sum()),
        "full_snapshot_roundtrip": "passed", "benchmark_results_match": performance["results_match"],
        "business_report": str((reports_dir / "PHASE3_SQL_BUSINESS_INSIGHTS.md").resolve()),
    }
    _write_json(output / "phase3_manifest.json", manifest)
    return manifest


def main() -> None:
    """Run all Phase 3 work, or execute a named query against a read-only database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-path", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--query", choices=list(QUERY_QUESTIONS))
    parser.add_argument("--params", help='JSON object, e.g. {"contract":"Two year","min_tenure":0,"max_tenure":12}')
    args = parser.parse_args()
    if args.params and not args.query:
        parser.error("--params requires --query")
    try:
        if args.query:
            parameters = json.loads(args.params) if args.params else None
            if parameters is not None and not isinstance(parameters, dict):
                raise ValueError("--params must be a JSON object")
            with closing(connect_database(args.database_path, readonly=True)) as conn:
                result = execute_query(conn, args.query, parameters)
            print(result.to_string(index=False))
        else:
            manifest = run_phase3(args.cleaned_path, args.database_path, args.reports_dir, benchmark_repeats=args.benchmark_repeats)
            print(json.dumps(manifest["sql_kpis"], indent=2))
            print(f'Inserted {manifest["ingestion"]["inserted_records"]:,} customers; executed {manifest["queries_executed"]} queries.')
            print(f'SQL/Pandas checks: {manifest["reconciliation_passed"]}/{manifest["reconciliation_checks"]}.')
            print(f'Business report: {manifest["business_report"]}')
    except (OSError, ValueError, RuntimeError, AssertionError, sqlite3.DatabaseError) as exc:
        parser.exit(1, f"Phase 3 failed: {exc}\n")


if __name__ == "__main__":
    main()
