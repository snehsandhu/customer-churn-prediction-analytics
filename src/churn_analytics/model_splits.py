"""Reproducible customer splits; Phase 4 can materialize train/validation only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DEVELOPMENT_ROLES = ("train", "validation")


def validate_target(frame: pd.DataFrame) -> pd.Series:
    """Check identity and the existing Yes/No -> 1/0 mapping without new cleaning."""
    required = {"customerID", "Churn", "ChurnFlag"}
    if not required.issubset(frame.columns) or not frame.columns.is_unique:
        raise ValueError("Missing/duplicate identity or target columns")
    ids = frame["customerID"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Customer IDs must be complete and unique")
    expected = frame["Churn"].map({"No": 0, "Yes": 1})
    target = frame["ChurnFlag"]
    if expected.isna().any() or not target.isin([0, 1]).all() or not target.eq(expected).all():
        raise ValueError("Target mapping must be Churn No=0, Yes=1 with no missing/other labels")
    return target.astype("int8")


def split_identity(manifest: dict) -> str:
    """Bind modeling operations to this fingerprint and these exact customer IDs."""
    payload = {"dataset_sha256": manifest["dataset_sha256"], "random_state": manifest["random_state"],
               "customer_ids": {role: manifest["splits"][role]["customer_ids"] for role in ("train", "validation", "test")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def create_split_manifest(frame: pd.DataFrame, dataset_sha256: str) -> dict:
    """Stratify 80/20, then 75/25 of development -> approximately 60/20/20.

    Sorting IDs first makes membership independent of CSV row order. Test labels
    are used only for required stratification/administrative class-count metadata;
    no test feature summaries, preprocessing, predictions or metrics are produced.
    """
    target = validate_target(frame)
    ordered = pd.DataFrame({"customerID": frame.customerID.astype(str), "target": target}).sort_values("customerID")
    if ordered.target.nunique() != 2:
        raise ValueError("Both target classes are required for stratified splitting")
    try:
        development, test = train_test_split(ordered, test_size=0.20, random_state=RANDOM_STATE, stratify=ordered.target)
        train, validation = train_test_split(development, test_size=0.25, random_state=RANDOM_STATE, stratify=development.target)
    except ValueError as exc:
        raise ValueError(f"Insufficient records per class for train/validation/test stratification: {exc}") from exc
    manifest = {"schema_version": 1, "dataset_sha256": dataset_sha256, "random_state": RANDOM_STATE,
                "method": "sort customerID; stratified 80/20, then stratified 75/25 of development; seed 42 for each split",
                "total_customers": len(frame), "splits": {}, "test_evaluated": False,
                "test_policy": "Reserved for Phase 5. Metadata/IDs only; no test transformations, fitting, predictions or metrics in Phase 4.",
                "prior_eda": "Phase 2 descriptively inspected the full dataset; this is not an entirely unseen holdout."}
    for role, subset in (("train", train), ("validation", validation), ("test", test)):
        manifest["splits"][role] = {
            "customer_ids": sorted(subset.customerID.tolist()), "size": len(subset),
            "class_counts": {str(label): int(subset.target.eq(label).sum()) for label in [0, 1]},
            "churn_proportion": float(subset.target.mean()),
        }
    manifest["manifest_id"] = split_identity(manifest)
    validate_split_manifest(frame, manifest, dataset_sha256)
    return manifest


def validate_split_manifest(frame: pd.DataFrame, manifest: dict, dataset_sha256: str) -> None:
    """Check complete coverage, disjointness, class metadata and source identity."""
    target = validate_target(frame)
    if manifest["dataset_sha256"] != dataset_sha256 or manifest["manifest_id"] != split_identity(manifest):
        raise ValueError("Split fingerprint/identity mismatch")
    if manifest["test_evaluated"] is not False or manifest["random_state"] != RANDOM_STATE:
        raise ValueError("Phase 4 requires seed 42 and a reserved, unevaluated test set")
    all_ids = []
    indexed_target = pd.Series(target.to_numpy(), index=frame.customerID.astype(str))
    for role in ("train", "validation", "test"):
        subset = manifest["splits"][role]
        ids = subset["customer_ids"]
        if len(ids) != len(set(ids)) or len(ids) != subset["size"]:
            raise ValueError(f"Duplicate IDs or invalid size in {role}")
        if not set(ids).issubset(set(indexed_target.index)):
            raise ValueError(f"Unknown customer IDs in {role}")
        labels = indexed_target.loc[ids]
        if any(int(labels.eq(k).sum()) != subset["class_counts"][str(k)] for k in [0, 1]):
            raise ValueError(f"Incorrect class counts in {role}")
        if not labels.nunique() == 2:
            raise ValueError(f"Both classes must occur in {role}")
        if abs(float(labels.mean()) - float(target.mean())) > 2 / len(labels):
            raise ValueError(f"Unexpected stratification difference in {role}")
        all_ids.extend(ids)
    if len(all_ids) != len(frame) or len(set(all_ids)) != len(frame) or set(all_ids) != set(indexed_target.index):
        raise ValueError("Splits must cover every customer exactly once with no overlap")
    if manifest["total_customers"] != len(frame):
        raise ValueError("Manifest total does not match the processed dataset")


@dataclass(frozen=True)
class DevelopmentSplit:
    """An explicitly labeled development partition; customer IDs are index labels."""
    role: str
    X: pd.DataFrame
    y: pd.Series


def development_split(frame: pd.DataFrame, manifest: dict, role: str) -> DevelopmentSplit:
    """Materialize only allowed rows, then select features/target separately."""
    if role not in DEVELOPMENT_ROLES:
        raise ValueError("Test split is reserved: Phase 4 permits only train/validation data")
    from .modeling import select_features
    ids = manifest["splits"][role]["customer_ids"]
    subset = frame.set_index("customerID", drop=False).loc[ids]
    target = validate_target(subset)
    return DevelopmentSplit(role=role, X=select_features(subset), y=target)
