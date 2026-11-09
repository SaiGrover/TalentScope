"""Single-employee inference with input validation and risk bands."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from talentscope import config
from talentscope.data import schema


@dataclass
class RiskResult:
    probability: float
    band: str
    threshold: float
    low_cut: float


def risk_band(p: float, threshold: float) -> str:
    """Documented bands relative to the selected threshold t: low < 0.5t <= elevated < t <= high."""
    low_cut = config.RISK_LOW_FRACTION * threshold
    if p >= threshold:
        return "high"
    if p >= low_cut:
        return "elevated"
    return "low"


def validate_employee(rec: dict) -> list[str]:
    """Return human-readable problems with a hand-entered record (empty list = valid)."""
    errors = []
    missing = [f for f in schema.MODEL_FEATURES if f not in rec or rec[f] is None or rec[f] == ""]
    if missing:
        errors.append(f"Missing values for: {', '.join(schema.PRETTY_NAMES.get(m, m) for m in missing)}.")
        return errors
    for col, allowed in schema.EXPECTED_CATEGORIES.items():
        if col in rec and str(rec[col]) not in allowed:
            errors.append(f"{schema.PRETTY_NAMES.get(col, col)} must be one of {allowed}.")
    numeric_ok = True
    for col, (lo, hi) in schema.VALUE_RANGES.items():
        if col in rec:
            try:
                v = float(rec[col])
            except (TypeError, ValueError):
                errors.append(f"{schema.PRETTY_NAMES.get(col, col)} must be a number.")
                numeric_ok = False
                continue
            if not lo <= v <= hi:
                errors.append(f"{schema.PRETTY_NAMES.get(col, col)} must be between {lo:g} and {hi:g}.")
                numeric_ok = False
    if numeric_ok:  # consistency rules need valid numbers, but run alongside category errors
        if rec["YearsAtCompany"] > rec["TotalWorkingYears"]:
            errors.append("Years at company cannot exceed total working years.")
        if rec["YearsInCurrentRole"] > rec["YearsAtCompany"]:
            errors.append("Years in current role cannot exceed years at company.")
        if rec["YearsWithCurrManager"] > rec["YearsAtCompany"]:
            errors.append("Years with current manager cannot exceed years at company.")
        if rec["YearsSinceLastPromotion"] > rec["YearsAtCompany"]:
            errors.append("Years since last promotion cannot exceed years at company.")
        if rec["TotalWorkingYears"] > rec["Age"] - 14:
            errors.append("Total working years is implausible for this age (career would start before 14).")
    return errors


def to_frame(rec: dict) -> pd.DataFrame:
    row = {f: rec[f] for f in schema.MODEL_FEATURES}
    df = pd.DataFrame([row])
    for c in schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES:
        df[c] = pd.to_numeric(df[c])
    for c in schema.NOMINAL_FEATURES:
        df[c] = df[c].astype(object)
    return df


def predict_employee(model, rec: dict, threshold: float) -> RiskResult:
    errors = validate_employee(rec)
    if errors:
        raise ValueError(" ".join(errors))
    p = float(model.predict_proba(to_frame(rec))[:, 1][0])
    return RiskResult(probability=p, band=risk_band(p, threshold), threshold=threshold,
                      low_cut=config.RISK_LOW_FRACTION * threshold)


def typical_employee(X: pd.DataFrame) -> dict:
    """Median of numeric features and mode of categorical ones (from the training split)."""
    rec = {}
    for c in schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES:
        rec[c] = int(np.round(X[c].median()))
    for c in schema.NOMINAL_FEATURES:
        rec[c] = str(X[c].mode().iloc[0])
    # roles belong to departments: take the most common role inside the modal department
    rec["JobRole"] = str(X.loc[X["Department"] == rec["Department"], "JobRole"].mode().iloc[0])
    return rec
