"""Measured index comparison on an in-memory database copy, never production data."""

from __future__ import annotations

from contextlib import closing
import math
import platform
import sqlite3
from statistics import median
from time import perf_counter_ns

import numpy as np

from .sql_analytics import query_parameters, read_query

INDEX_NAME = "idx_snapshot_contract_tenure_charges"


def benchmark_index(conn: sqlite3.Connection, repeats: int = 100) -> dict:
    """Measure a real filtered summary before/after one covering index.

    The copy is memory-resident and warmed before each block. Timings include
    Python execute/fetch and SQL preparation (statement caching disabled), exclude
    index creation/copying, and are sequential blocks, not a controlled hardware
    study. Small-dataset differences can be noisy or reverse on other workloads.
    """
    if type(repeats) is not int or not 1 <= repeats <= 10000:
        raise ValueError("Benchmark repeats must be an integer between 1 and 10000")
    name = "20_filtered_contract_tenure"
    query, params = read_query(name), query_parameters(name)
    ddl_row = conn.execute("SELECT sql FROM sqlite_schema WHERE type='index' AND name=?", (INDEX_NAME,)).fetchone()
    if ddl_row is None:
        raise ValueError("Expected covering index is missing; run database ingestion first")
    with closing(sqlite3.connect(":memory:", cached_statements=0)) as scratch:
        conn.backup(scratch)
        scratch.execute("PRAGMA automatic_index = OFF")
        scratch.execute(f"DROP INDEX {INDEX_NAME}")
        scratch.execute("ANALYZE")
        def run_block():
            plan = [dict(zip(["id", "parent", "unused", "detail"], row)) for row in scratch.execute("EXPLAIN QUERY PLAN " + query, params)]
            for _ in range(5):
                scratch.execute(query, params).fetchall()
            elapsed = []
            for _ in range(repeats):
                start = perf_counter_ns()
                result = scratch.execute(query, params).fetchall()
                elapsed.append((perf_counter_ns() - start) / 1_000_000)
            return {"plan": plan, "median_ms": median(elapsed), "minimum_ms": min(elapsed),
                    "p95_ms": float(np.percentile(elapsed, 95)), "repeats": repeats}, result
        before, before_rows = run_block()
        scratch.execute(ddl_row[0])
        scratch.execute("ANALYZE")
        after, after_rows = run_block()
        equal = len(before_rows) == len(after_rows)
        if equal:
            for left, right in zip(before_rows, after_rows):
                for a, b in zip(left, right):
                    equal &= (a is None and b is None) if a is None or b is None else math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
        if not equal:
            raise ValueError("Indexed and unindexed query results differ")
    return {"query": name, "parameters": params, "sqlite_version": sqlite3.sqlite_version,
            "python_version": platform.python_version(), "database_customers": conn.execute("SELECT COUNT(*) FROM customer_snapshot").fetchone()[0],
            "before_index": before, "after_index": after, "results_match": bool(equal),
            "result_rows": before_rows, "observed_before_over_after_median": before["median_ms"] / after["median_ms"],
            "method": "In-memory copy; five warmups/block; no statement cache; sequential unindexed/indexed blocks; excludes index creation; automatic indexes disabled."}
