"""Cached data and model access shared by every page.

Expensive objects are cached once per server process (``st.cache_data`` for
frames, ``st.cache_resource`` for models and explainers). Cheap filtering is
done outside the cache on every rerun.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from talentscope import config
from talentscope.analytics import apply_filters
from talentscope.data import load_dataset, load_raw, split_features_target, validate_dataset
from talentscope.modeling.artifacts import list_runs, load_bundle
from talentscope.preprocessing import make_split


@st.cache_data(show_spinner=False)
def dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def raw_dataset() -> pd.DataFrame:
    return load_raw()


@st.cache_resource(show_spinner=False)
def validation_report():
    return validate_dataset(load_raw())


@st.cache_data(show_spinner=False)
def split_frames(seed: int = config.RANDOM_SEED, test_size: float = config.TEST_SIZE):
    df = dataset()
    X, y = split_features_target(df)
    return make_split(X, y, test_size, seed)


def run_dirs() -> list[Path]:
    return list_runs()


def _mtime(p: Path) -> float:
    f = p / "selection.json"
    return f.stat().st_mtime if f.exists() else 0.0


@st.cache_resource(show_spinner=False, max_entries=8)
def _bundle(run_dir: str, mtime: float):
    return load_bundle(Path(run_dir))


def bundle(run_dir: Path | None = None):
    """The active experiment bundle, or None when no trained model exists yet."""
    run_dir = Path(run_dir) if run_dir else active_run()
    if run_dir is None:
        return None
    try:
        return _bundle(str(run_dir), _mtime(run_dir))
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False, max_entries=8)
def _model(model_path: str, mtime: float):
    import joblib
    return joblib.load(model_path)


def model(b):
    return _model(str(b.model_path), b.model_path.stat().st_mtime)


def active_run() -> Path | None:
    runs = run_dirs()
    if not runs:
        return None
    chosen = st.session_state.get("active_run")
    if chosen and Path(chosen) in runs:
        return Path(chosen)
    return runs[0]


def run_label(p: Path) -> str:
    if p.resolve() == config.DEFAULT_EXPERIMENT_DIR.resolve():
        return "Default experiment (committed)"
    return f"Lab run {p.name}"


def run_picker(key: str) -> Path | None:
    """Select which experiment the model pages read; shared across pages."""
    runs = run_dirs()
    if not runs:
        return None
    current = active_run()
    idx = runs.index(current) if current in runs else 0
    choice = st.selectbox("Experiment", runs, index=idx, format_func=run_label, key=key,
                          help="The committed default run, or a run you launched in the Model Laboratory.")
    st.session_state["active_run"] = str(choice)
    return choice


# ---------------------------------------------------------------- shared filters
FILTER_KEYS = {"f_department": "Department", "f_role": "JobRole", "f_gender": "Gender", "f_overtime": "OverTime"}


def filter_bar(df: pd.DataFrame, extended: bool = False, key: str = "filters") -> pd.DataFrame:
    """One filter row above everything it scopes. Values persist across pages."""
    with st.container(key=key):
        cols = st.columns([1.2, 1.6, 0.9, 0.9, 1.3] if extended else [1.2, 1.8, 1], vertical_alignment="bottom")
        dept = cols[0].multiselect("Department", sorted(df["Department"].unique()), key="f_department",
                                   persist_state="session", placeholder="All departments")
        roles_all = sorted(df.loc[df["Department"].isin(dept) if dept else slice(None), "JobRole"].unique())
        role = cols[1].multiselect("Job role", roles_all, key="f_role", persist_state="session",
                                   placeholder="All roles")
        if extended:
            gender = cols[2].multiselect("Gender", sorted(df["Gender"].unique()), key="f_gender",
                                         persist_state="session", placeholder="All")
            ot = cols[3].multiselect("Overtime", ["Yes", "No"], key="f_overtime", persist_state="session",
                                     placeholder="All")
            age = cols[4].slider("Age", int(df["Age"].min()), int(df["Age"].max()),
                                 (int(df["Age"].min()), int(df["Age"].max())), key="f_age", persist_state="session")
        else:
            gender, ot, age = [], [], None
            n_active = sum(bool(st.session_state.get(k)) for k in ("f_gender", "f_overtime"))
            cols[2].caption(f"{n_active} more filter(s) active from Workforce Explorer" if n_active else
                            "More filters on Workforce Explorer")
            gender = st.session_state.get("f_gender") or []
            ot = st.session_state.get("f_overtime") or []
            age = st.session_state.get("f_age")
    filters = {"Department": dept, "JobRole": [r for r in role if r in roles_all], "Gender": gender, "OverTime": ot}
    if age:
        filters["Age"] = tuple(age)
    out = apply_filters(df, filters)
    return out


def filter_summary(df_all: pd.DataFrame, df: pd.DataFrame) -> str:
    if len(df) == len(df_all):
        return f"All {len(df_all):,} employees"
    return f"{len(df):,} of {len(df_all):,} employees match the filters"
