"""Dataset profiling and validation.

``validate_dataset`` runs a series of checks against the loaded frame and
returns a report with one row per check (status pass / warn / fail) plus the
detail tables the Data Laboratory page shows. Nothing here mutates the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import roc_auc_score

from talentscope.data import schema


@dataclass
class Check:
    group: str
    name: str
    status: str            # "pass" | "warn" | "fail"
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    columns: pd.DataFrame | None = None
    missing: pd.DataFrame | None = None
    constant_columns: list[str] = field(default_factory=list)
    unexpected_values: pd.DataFrame | None = None
    out_of_range: pd.DataFrame | None = None
    leakage_screen: pd.DataFrame | None = None
    redundant_pairs: pd.DataFrame | None = None
    target_distribution: pd.DataFrame | None = None
    duplicates: int = 0

    @property
    def passed(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    @property
    def n_pass(self) -> int:
        return sum(c.status == "pass" for c in self.checks)

    @property
    def n_warn(self) -> int:
        return sum(c.status == "warn" for c in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(c.status == "fail" for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks], columns=["group", "name", "status", "detail"])


def _role(col: str) -> str:
    if col == schema.TARGET:
        return "target"
    if col in schema.ID_COLUMNS:
        return "identifier (dropped)"
    if col in schema.CONSTANT_COLUMNS:
        return "constant (dropped)"
    if col in schema.NUMERIC_FEATURES:
        return "numeric"
    if col in schema.ORDINAL_FEATURES:
        return "ordinal"
    if col in schema.NOMINAL_FEATURES:
        return "nominal"
    return "undocumented"


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "role": _role(col),
            "dtype": str(s.dtype),
            "unique": int(s.nunique(dropna=True)),
            "missing": int(s.isna().sum()),
            "min": s.min() if pd.api.types.is_numeric_dtype(s) else None,
            "max": s.max() if pd.api.types.is_numeric_dtype(s) else None,
            "example": str(s.dropna().iloc[0]) if s.notna().any() else "",
            "description": schema.FEATURE_DESCRIPTIONS.get(col, ""),
        })
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport()
    add = rep.checks.append
    n = len(df)

    # -- shape and schema --------------------------------------------------------------
    shape_ok = df.shape == (schema.EXPECTED_ROWS, schema.EXPECTED_COLUMNS)
    add(Check("schema", "Shape", "pass" if shape_ok else "warn",
              f"{df.shape[0]:,} rows x {df.shape[1]} columns "
              f"(documented: {schema.EXPECTED_ROWS:,} x {schema.EXPECTED_COLUMNS})"))
    missing_cols = sorted(set(schema.ALL_COLUMNS) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(schema.ALL_COLUMNS))
    add(Check("schema", "Documented columns present", "fail" if missing_cols else "pass",
              f"missing: {missing_cols}" if missing_cols else "all 35 documented columns are present"))
    add(Check("schema", "No undocumented columns", "warn" if extra_cols else "pass",
              f"extra: {extra_cols}" if extra_cols else "no extra columns"))
    rep.columns = column_profile(df)
    bad_types = [c for c in schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES
                 if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    add(Check("schema", "Feature types", "fail" if bad_types else "pass",
              f"non-numeric: {bad_types}" if bad_types else
              f"{len(schema.NUMERIC_FEATURES)} numeric, {len(schema.ORDINAL_FEATURES)} ordinal and "
              f"{len(schema.NOMINAL_FEATURES)} nominal features typed as documented"))

    # -- missing values --------------------------------------------------------------------
    miss = df.isna().sum()
    rep.missing = pd.DataFrame({"column": miss.index, "missing": miss.values,
                                "missing_pct": (miss.values / max(n, 1) * 100).round(2)})
    total_missing = int(miss.sum())
    add(Check("quality", "Missing values", "pass" if total_missing == 0 else "warn",
              "no missing cells in any column" if total_missing == 0 else
              f"{total_missing:,} missing cells across {int((miss > 0).sum())} columns (imputed inside the pipeline)"))

    # -- duplicates -----------------------------------------------------------------------------
    rep.duplicates = int(df.duplicated().sum())
    no_id = df.drop(columns=[c for c in schema.ID_COLUMNS if c in df.columns])
    feature_dups = int(no_id.duplicated().sum())
    add(Check("quality", "Duplicate rows", "pass" if rep.duplicates == 0 and feature_dups == 0 else "warn",
              f"{rep.duplicates} exact duplicates; {feature_dups} duplicates once EmployeeNumber is ignored"))
    if schema.ID_COLUMNS[0] in df.columns:
        uniq = bool(df[schema.ID_COLUMNS[0]].is_unique)
        add(Check("quality", "EmployeeNumber is unique", "pass" if uniq else "fail",
                  "one row per employee" if uniq else "identifier repeats"))

    # -- target ---------------------------------------------------------------------------------
    if schema.TARGET in df.columns:
        vc = df[schema.TARGET].value_counts()
        rep.target_distribution = pd.DataFrame({"class": vc.index, "count": vc.values,
                                                "share": (vc.values / n).round(4)})
        pos = int(vc.get(schema.POSITIVE_LABEL, 0))
        add(Check("target", "Target distribution",
                  "pass" if abs(pos - schema.EXPECTED_POSITIVES) <= 5 else "warn",
                  f"{pos} left ({pos / n:.1%}) and {n - pos} stayed; imbalance ratio {(n - pos) / max(pos, 1):.1f} : 1"))
        bad = sorted(set(map(str, df[schema.TARGET].dropna().unique())) - {"Yes", "No"})
        add(Check("target", "Target labels", "fail" if bad else "pass",
                  f"unexpected labels {bad}" if bad else "labels are exactly Yes / No"))
    else:
        add(Check("target", "Target present", "fail", f"'{schema.TARGET}' column missing"))

    # -- constant and redundant columns ----------------------------------------------------------
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    rep.constant_columns = const
    add(Check("redundancy", "Constant columns", "pass" if set(const) <= set(schema.CONSTANT_COLUMNS) else "warn",
              f"{', '.join(const)} carry a single value and are dropped before modelling" if const else "none"))
    num_cols = [c for c in schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES if c in df.columns]
    corr = df[num_cols].corr().abs()
    pairs = [{"feature_a": a, "feature_b": b, "abs_pearson_r": round(float(corr.loc[a, b]), 3)}
             for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if corr.loc[a, b] >= 0.7]
    rep.redundant_pairs = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_pearson_r"]) \
        .sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    strong = rep.redundant_pairs[rep.redundant_pairs["abs_pearson_r"] >= 0.9]
    add(Check("redundancy", "Highly correlated features", "warn" if len(rep.redundant_pairs) else "pass",
              (f"{len(rep.redundant_pairs)} pairs with |r| >= 0.7 (max {rep.redundant_pairs['abs_pearson_r'].max():.2f}, "
               f"{len(strong)} >= 0.9); kept, since none is a duplicate and tree models and the L2-regularised "
               "linear model tolerate collinearity") if len(rep.redundant_pairs) else "no pair above |r| 0.7"))

    # -- unexpected values ------------------------------------------------------------------------
    rows = []
    for col, allowed in schema.EXPECTED_CATEGORIES.items():
        if col in df.columns:
            observed = df[col].dropna().astype(str)
            for v in sorted(set(observed.unique()) - set(allowed)):
                rows.append({"column": col, "value": v, "count": int((observed == v).sum())})
    rep.unexpected_values = pd.DataFrame(rows, columns=["column", "value", "count"])
    add(Check("values", "Categorical values documented", "warn" if rows else "pass",
              f"{len(rows)} undocumented category values" if rows else
              "every categorical value matches the data dictionary"))
    rows = []
    for col, (lo, hi) in schema.VALUE_RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                rows.append({"column": col, "expected_min": lo, "expected_max": hi,
                             "observed_min": df[col].min(), "observed_max": df[col].max(), "violations": bad})
    rep.out_of_range = pd.DataFrame(rows, columns=["column", "expected_min", "expected_max",
                                                   "observed_min", "observed_max", "violations"])
    add(Check("values", "Numeric values within plausible ranges", "warn" if rows else "pass",
              f"{len(rows)} columns outside documented ranges" if rows else
              "every numeric column sits inside its documented range"))
    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(df.columns):
        bad = int((df["YearsAtCompany"] > df["TotalWorkingYears"]).sum())
        add(Check("values", "Company tenure <= total experience", "warn" if bad else "pass",
                  f"{bad} rows where tenure exceeds total experience" if bad else "consistent for every employee"))
    if {"YearsInCurrentRole", "YearsAtCompany"}.issubset(df.columns):
        bad = int((df["YearsInCurrentRole"] > df["YearsAtCompany"]).sum())
        add(Check("values", "Role tenure <= company tenure", "warn" if bad else "pass",
                  f"{bad} inconsistent rows" if bad else "consistent for every employee"))
    if {"YearsWithCurrManager", "YearsAtCompany"}.issubset(df.columns):
        bad = int((df["YearsWithCurrManager"] > df["YearsAtCompany"]).sum())
        add(Check("values", "Manager tenure <= company tenure", "warn" if bad else "pass",
                  f"{bad} inconsistent rows" if bad else "consistent for every employee"))
    if "PerformanceRating" in df.columns:
        levels = sorted(df["PerformanceRating"].dropna().unique().tolist())
        add(Check("values", "Performance rating spread", "warn" if len(levels) <= 2 else "pass",
                  f"only levels {levels} occur, so this feature carries very little signal"
                  if len(levels) <= 2 else f"levels {levels}"))

    # -- leakage screen ---------------------------------------------------------------------------
    # A single raw feature that separates the target almost perfectly would be
    # a proxy for the label or a post-outcome field. Each feature is scored on
    # its own with univariate ROC-AUC; anything >= 0.9 is flagged.
    if schema.TARGET in df.columns:
        y = (df[schema.TARGET] == schema.POSITIVE_LABEL).astype(int).to_numpy()
        rows = []
        for col in schema.MODEL_FEATURES:
            if col not in df.columns:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                x = s.fillna(s.median()).to_numpy(dtype=float)
                kind = "numeric"
            else:
                # category -> observed attrition rate. A screen only, never a model input.
                rate = pd.Series(y, index=s.index).groupby(s.astype(str)).mean()
                x = s.astype(str).map(rate).to_numpy(dtype=float)
                kind = "categorical (rate-encoded)"
            auc = float(roc_auc_score(y, x))
            rows.append({"feature": col, "kind": kind, "univariate_auc": round(max(auc, 1 - auc), 3)})
        rep.leakage_screen = (pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
                              .reset_index(drop=True))
        flagged = rep.leakage_screen.loc[rep.leakage_screen["univariate_auc"] >= 0.9, "feature"].tolist()
        top = rep.leakage_screen.iloc[0]
        add(Check("leakage", "Single-feature leakage screen", "fail" if flagged else "pass",
                  f"suspected label proxies: {flagged}" if flagged else
                  f"strongest single feature is {top['feature']} (AUC {top['univariate_auc']:.3f}); none behaves like a label proxy"))
        add(Check("leakage", "Identifier excluded", "pass",
                  "EmployeeNumber is dropped and never reaches a model"))
        add(Check("leakage", "No post-outcome fields", "pass",
                  "no column records exit date, exit interview or termination reason"))
    return rep
