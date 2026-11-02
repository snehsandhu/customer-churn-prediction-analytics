"""Reuse persisted membership; never materialize held-out fields for development."""
from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path

import pandas as pd

from .cleaning import load_cleaned_dataset
from .model_splits import RANDOM_STATE, development_split, split_identity


def sha256(path: str | Path) -> str:
    """Fingerprint bytes without changing the file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict:
    """Load existing structured evidence."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path: str | Path, value: dict) -> None:
    """Create evidence exclusively: reruns cannot silently replace it."""
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def load_membership(root: str | Path) -> dict:
    """Check fingerprint, unique/disjoint complete IDs, seed and stored assignment.

    Only the customerID CSV column is parsed here. Test labels remain inaccessible
    until the final evaluation claim is durably created by final_evaluation.py.
    """
    root = Path(root)
    manifest = read_json(root / "reports/phase4/split_manifest.json")
    source = root / "data/processed/telco_churn_cleaned.csv"
    if manifest["dataset_sha256"] != sha256(source) or manifest["manifest_id"] != split_identity(manifest):
        raise ValueError("Dataset/split fingerprint mismatch")
    if manifest["random_state"] != RANDOM_STATE or manifest["test_evaluated"] is not False:
        raise ValueError("Expected the unchanged historical Phase 4 reserved split")
    ids = pd.read_csv(source, usecols=["customerID"], dtype="string").customerID
    if ids.isna().any() or ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError("Missing/duplicate dataset customer IDs")
    combined = []
    for role in ("train", "validation", "test"):
        info = manifest["splits"][role]
        names = info["customer_ids"]
        if len(names) != info["size"] or len(set(names)) != len(names):
            raise ValueError(f"Duplicate IDs/incorrect size in {role}")
        if sum(info["class_counts"].values()) != len(names):
            raise ValueError("Invalid class-count metadata")
        combined.extend(names)
    if len(combined) != len(set(combined)) or set(combined) != set(ids) or len(ids) != manifest["total_customers"]:
        raise ValueError("Incomplete/overlapping split membership")
    assignments = pd.read_csv(root / "reports/phase4/split_assignments.csv", dtype="string")
    expected = {(identifier, role) for role, info in manifest["splits"].items() for identifier in info["customer_ids"]}
    if len(assignments) != len(expected) or set(map(tuple, assignments[["customerID", "split"]].to_numpy())) != expected:
        raise ValueError("Stored split assignments differ from manifest")
    return manifest


def _read_selected_rows(source: str | Path, ids: list[str]) -> pd.DataFrame:
    """Admit lines by their first ID token before parsing the other CSV fields.

    This source-specific reader intentionally fails for quoted/unexpected IBM ID
    formats rather than falling back to parsing all reserved labels. It supports
    the existing IBM file and ordinary synthetic IDs used in tests.
    """
    wanted = set(ids)
    if len(wanted) != len(ids):
        raise ValueError("Duplicate requested IDs")
    lines, seen = [], set()
    with Path(source).open(encoding="utf-8", newline="") as handle:
        lines.append(next(handle))
        for line in handle:
            identifier = line.partition(",")[0]
            if identifier in wanted:
                if identifier in seen:
                    raise ValueError("Duplicate source customer ID")
                seen.add(identifier)
                lines.append(line)
    if seen != wanted:
        raise ValueError("Requested customers not found in source")
    return load_cleaned_dataset(StringIO("".join(lines)))


def load_development(root: str | Path, manifest: dict):
    """Return only guarded original training and validation partitions."""
    source = Path(root) / "data/processed/telco_churn_cleaned.csv"
    ids = manifest["splits"]["train"]["customer_ids"] + manifest["splits"]["validation"]["customer_ids"]
    frame = _read_selected_rows(source, ids)
    subsets = [development_split(frame, manifest, role) for role in ("train", "validation")]
    for subset in subsets:
        expected = manifest["splits"][subset.role]["class_counts"]
        if {str(k): int(subset.y.eq(k).sum()) for k in [0, 1]} != expected:
            raise ValueError("Development class counts differ from persisted metadata")
    return tuple(subsets)
