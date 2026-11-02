"""Run Phase 2 locally: cleaning, auditable tables, figures and business report."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import (
    ASSOCIATION_FEATURES, CATEGORICAL_FEATURES, association_tests, categorical_summary,
    churn_by_segment, correlation_tables, customer_kpis, numerical_summary,
)
from .cleaning import (
    DEFAULT_CLEAN_PATH, build_quality_report, clean_dataset,
    load_cleaned_dataset, save_cleaned_dataset,
)
from .data import DEFAULT_RAW_PATH, PROJECT_ROOT, load_dataset
from .reporting import business_insights_markdown, quality_markdown

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _json_safe(value):
    """Encode undefined statistics as JSON null rather than nonstandard NaN."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value) -> None:
    """Write strict, human-readable JSON for downstream checks."""
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_phase2(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    cleaned_path: str | Path = DEFAULT_CLEAN_PATH,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict:
    """Run a reproducible workflow, protecting source bytes and validating reload.

    No network calls are made. Generated analytical outputs are refreshed on
    reruns. Cleaning never learns parameters from the target or complete data.
    """
    raw_path, cleaned_path, reports_dir = map(Path, (raw_path, cleaned_path, reports_dir))
    if raw_path.resolve() == cleaned_path.resolve():
        raise ValueError("Raw and processed paths must differ")
    if reports_dir.resolve().is_relative_to(DEFAULT_RAW_PATH.parent.resolve()):
        raise ValueError("Reports must not be written into data/raw/")
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw = load_dataset(raw_path)
    cleaned = clean_dataset(raw)
    quality = build_quality_report(raw, cleaned)
    save_cleaned_dataset(cleaned, cleaned_path)
    reloaded = load_cleaned_dataset(cleaned_path)
    pd.testing.assert_frame_equal(cleaned.reset_index(drop=True), reloaded, check_exact=False, rtol=1e-12, atol=1e-12)
    # The saved/reloaded artifact is the common input to every analytical output.
    cleaned = reloaded
    kpis = customer_kpis(cleaned)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    tests = association_tests(cleaned)
    tables = {
        "numerical_summary": numerical_summary(cleaned),
        "numerical_summary_by_churn": numerical_summary(cleaned, by_churn=True),
        "categorical_summary": categorical_summary(cleaned),
        "association_tests": tests,
    }
    for feature in (*CATEGORICAL_FEATURES, "TenureGroup", "MonthlyChargeGroup"):
        tables[f"churn_by_{feature}"] = churn_by_segment(cleaned, feature)
    tables["segment_contract_tenure"] = churn_by_segment(cleaned, ["Contract", "TenureGroup"])
    tables["segment_internet_support"] = churn_by_segment(cleaned, ["InternetService", "TechSupport"])
    tables["segment_contract_monthly_charges"] = churn_by_segment(cleaned, ["Contract", "MonthlyChargeGroup"])
    tables["security_internet_subscribers"] = churn_by_segment(cleaned.loc[cleaned.InternetService.ne("No")], "OnlineSecurity")
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    for name, table in correlation_tables(cleaned).items():
        table.to_csv(tables_dir / f"correlation_{name}.csv", index=True)
    _write_json(reports_dir / "kpis.json", kpis)
    _write_json(reports_dir / "data_quality_report.json", quality)
    (reports_dir / "data_quality_report.md").write_text(quality_markdown(quality), encoding="utf-8")
    # Batch rendering stays independent of graphical displays or a running browser.
    import matplotlib
    matplotlib.use("Agg")
    from .plots import create_figures
    figures = create_figures(cleaned, reports_dir / "figures")
    (reports_dir / "PHASE2_BUSINESS_INSIGHTS.md").write_text(
        business_insights_markdown(cleaned, kpis, quality, tests), encoding="utf-8",
    )
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Raw source fingerprint changed during Phase 2")
    manifest = {
        "raw_sha256": source_hash, "cleaned_sha256": hashlib.sha256(cleaned_path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "cleaned_rows": len(cleaned), "cleaned_columns": cleaned.shape[1],
        "processed_file": str(cleaned_path.resolve()), "figures": [str(p.resolve()) for p in figures],
        "table_files": [str(p.resolve()) for p in sorted(tables_dir.glob("*.csv"))],
        "association_family": list(ASSOCIATION_FEATURES),
        "package_versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "matplotlib", "seaborn"]},
        "kpis": kpis,
    }
    _write_json(reports_dir / "phase2_manifest.json", manifest)
    return manifest


def main() -> None:
    """Run Phase 2 from any directory and print the actual customer KPIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--cleaned-path", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()
    try:
        manifest = run_phase2(args.raw_path, args.cleaned_path, args.reports_dir)
    except (OSError, ValueError, AssertionError) as exc:
        parser.exit(1, f"Phase 2 failed: {exc}\n")
    print(json.dumps(_json_safe(manifest["kpis"]), indent=2, allow_nan=False))
    print(f"Generated {len(manifest['figures'])} figures and {len(manifest['table_files'])} tables.")
    print(f"Business insights: {(args.reports_dir / 'PHASE2_BUSINESS_INSIGHTS.md').resolve()}")


if __name__ == "__main__":
    main()
