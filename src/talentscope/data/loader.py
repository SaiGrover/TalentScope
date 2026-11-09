"""Load the IBM HR attrition dataset and derive the analysis frame.

The raw CSV is committed under ``data/raw``; nothing is downloaded at run time
and no rows are ever generated. ``prepare_dataset`` drops the identifier and
constant columns and adds a 0/1 target flag; ``add_analysis_bands`` adds the
age, income and tenure bands used by the analytics pages.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from talentscope import __version__, config
from talentscope.data import schema

AGE_BINS = [17, 25, 35, 45, 55, 65]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56+"]
INCOME_BINS = [0, 3000, 5000, 8000, 12000, 20001]
INCOME_LABELS = ["<3k", "3k-5k", "5k-8k", "8k-12k", "12k+"]
TENURE_BINS = [-1, 1, 3, 5, 10, 20, 45]
TENURE_LABELS = ["0-1", "2-3", "4-5", "6-10", "11-20", "21+"]
DISTANCE_BINS = [0, 5, 10, 20, 30]
DISTANCE_LABELS = ["1-5", "6-10", "11-20", "21-29"]
PROMOTION_BINS = [-1, 0, 2, 5, 45]
PROMOTION_LABELS = ["This year", "1-2 yrs ago", "3-5 yrs ago", "6+ yrs ago"]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the committed CSV exactly as shipped (35 columns)."""
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected the IBM HR attrition CSV; see data/README.md."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    # pandas 3 reads text as the 'str' dtype; normalise to object for scikit-learn and SMOTENC
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object)
    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/constant columns and add the binary target flag.

    The string ``Attrition`` column is kept for readable charts; ``AttritionFlag``
    is the modelling target (1 = left).
    """
    out = df.drop(columns=[c for c in schema.DROP_COLUMNS if c in df.columns]).copy()
    out[schema.TARGET_FLAG] = (out[schema.TARGET] == schema.POSITIVE_LABEL).astype(int)
    return out


def add_analysis_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical bands used by the workforce and attrition analytics."""
    out = df.copy()
    out["AgeBand"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    out["IncomeBand"] = pd.cut(out["MonthlyIncome"], bins=INCOME_BINS, labels=INCOME_LABELS)
    out["TenureBand"] = pd.cut(out["YearsAtCompany"], bins=TENURE_BINS, labels=TENURE_LABELS)
    out["DistanceBand"] = pd.cut(out["DistanceFromHome"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS)
    out["PromotionRecency"] = pd.cut(out["YearsSinceLastPromotion"], bins=PROMOTION_BINS,
                                     labels=PROMOTION_LABELS)
    for col, levels in schema.ORDINAL_LEVELS.items():
        out[f"{col}Label"] = pd.Categorical(
            out[col].map(levels), categories=list(levels.values()), ordered=True
        )
    return out


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Raw -> prepared -> banded: the frame every analytics page works from."""
    return add_analysis_bands(prepare_dataset(load_raw(path)))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the modelling matrix (30 features) and the 0/1 target."""
    X = df[schema.MODEL_FEATURES].copy()
    if schema.TARGET_FLAG in df.columns:
        y = df[schema.TARGET_FLAG].astype(int)
    else:
        y = (df[schema.TARGET] == schema.POSITIVE_LABEL).astype(int)
    y.name = schema.TARGET_FLAG
    return X, y


def dataset_fingerprint(path: str | Path | None = None) -> str:
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_card(df: pd.DataFrame | None = None, path: str | Path | None = None) -> dict:
    """Describe the dataset version actually on disk (used by model cards and reports)."""
    path = Path(path) if path is not None else config.RAW_DATA_PATH
    if df is None:
        df = load_raw(path)
    pos = int((df[schema.TARGET] == schema.POSITIVE_LABEL).sum())
    try:
        rel = path.resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        rel = str(path)
    return {
        "name": config.DATASET_NAME,
        "source_url": config.DATASET_SOURCE_URL,
        "mirror_url": config.DATASET_MIRROR_URL,
        "license": config.DATASET_LICENSE,
        "synthetic": True,
        "file": rel,
        "sha256": dataset_fingerprint(path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "positives": pos,
        "attrition_rate": round(pos / len(df), 4),
        "talentscope_version": __version__,
    }
