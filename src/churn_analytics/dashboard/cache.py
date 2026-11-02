"""Cache immutable public analytics and the trusted model, never user uploads."""
from pathlib import Path
import streamlit as st
from ..data import PROJECT_ROOT
from .artifacts import MODEL_PATH, verify_final_bundle
from .data import load_analytics
from .prediction import load_predictor


def signature(paths) -> tuple:
    """Invalidate caches when known local artifact metadata changes."""
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else None for p in paths)


@st.cache_data(show_spinner=False, max_entries=2)
def _analytics(root: str, identity: tuple):
    """Cache copies of the validated public snapshot without customer IDs."""
    return load_analytics(Path(root))


def analytics_frame():
    """Resolve paths relative to the package root, not the current directory."""
    paths = [PROJECT_ROOT/"data/processed/telco_churn_cleaned.csv", PROJECT_ROOT/"reports/phase4/split_manifest.json"]
    return _analytics(str(PROJECT_ROOT), signature(paths))


@st.cache_resource(show_spinner=False, max_entries=2)
def _predictor(root: str, identity: tuple):
    """Retain a read-only fitted estimator; a cache miss never invokes fitting."""
    return load_predictor(Path(root))


def predictor_resource():
    """Verify evidence on each scoring request, including when resource is cached."""
    verify_final_bundle(PROJECT_ROOT)
    paths = [PROJECT_ROOT/MODEL_PATH, *[PROJECT_ROOT/f"reports/phase5/{name}.json" for name in
        ("final_model_metadata", "final_selection_manifest", "test_consumption", "final_test_metrics")]]
    return _predictor(str(PROJECT_ROOT), signature(paths))
