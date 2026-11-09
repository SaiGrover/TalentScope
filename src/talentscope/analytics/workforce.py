"""Workforce and attrition analytics.

All functions take the prepared frame (see ``talentscope.data.load_dataset``)
and return plain DataFrames or dicts, so the dashboard, the notebooks and the
reports compute identical numbers. Group rates carry Wilson 95% intervals and
group sizes; association tests carry Benjamini-Hochberg adjusted p-values.
Every finding here is descriptive: an association in synthetic data, not a cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from talentscope.data import schema
from talentscope.evaluation.metrics import wilson_interval

TARGET = schema.TARGET_FLAG

# Analysis dimensions: (column in the prepared frame, readable label, ordered?)
DIMENSIONS: dict[str, tuple[str, str, bool]] = {
    "department": ("Department", "Department", False),
    "job_role": ("JobRole", "Job role", False),
    "overtime": ("OverTime", "Overtime", False),
    "age": ("AgeBand", "Age band", True),
    "income": ("IncomeBand", "Monthly income band", True),
    "tenure": ("TenureBand", "Years at company", True),
    "job_satisfaction": ("JobSatisfactionLabel", "Job satisfaction", True),
    "environment": ("EnvironmentSatisfactionLabel", "Environment satisfaction", True),
    "work_life": ("WorkLifeBalanceLabel", "Work-life balance", True),
    "involvement": ("JobInvolvementLabel", "Job involvement", True),
    "travel": ("BusinessTravel", "Business travel", False),
    "promotion": ("PromotionRecency", "Last promotion", True),
    "job_level": ("JobLevelLabel", "Job level", True),
    "stock": ("StockOptionLevelLabel", "Stock option level", True),
    "marital": ("MaritalStatus", "Marital status", False),
    "education": ("EducationLabel", "Education", True),
    "education_field": ("EducationField", "Education field", False),
    "gender": ("Gender", "Gender", False),
    "distance": ("DistanceBand", "Commute distance", True),
}


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Filter by {column: [allowed values]} and {column: (lo, hi)} ranges. Empty lists mean 'all'."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns or val is None:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            mask &= df[col].between(val[0], val[1])
        elif isinstance(val, (list, set)) and len(val) > 0:
            mask &= df[col].astype(str).isin([str(v) for v in val])
    return df[mask]


def workforce_kpis(df: pd.DataFrame) -> dict:
    n = int(len(df))
    left = int(df[TARGET].sum()) if n else 0
    lo, hi = wilson_interval(left, n) if n else (np.nan, np.nan)
    return {
        "headcount": n,
        "left": left,
        "stayed": n - left,
        "attrition_rate": left / n if n else np.nan,
        "rate_ci": (lo, hi),
        "avg_tenure": float(df["YearsAtCompany"].mean()) if n else np.nan,
        "median_tenure": float(df["YearsAtCompany"].median()) if n else np.nan,
        "avg_income": float(df["MonthlyIncome"].mean()) if n else np.nan,
        "median_income": float(df["MonthlyIncome"].median()) if n else np.nan,
        "avg_age": float(df["Age"].mean()) if n else np.nan,
        "overtime_share": float((df["OverTime"] == "Yes").mean()) if n else np.nan,
        "departments": int(df["Department"].nunique()) if n else 0,
        "roles": int(df["JobRole"].nunique()) if n else 0,
    }


def rate_by(df: pd.DataFrame, col: str, min_n: int = 1, sort: str | None = None) -> pd.DataFrame:
    """Attrition rate per group with counts, Wilson 95% CI and lift over the filtered overall rate."""
    if df.empty:
        return pd.DataFrame(columns=["group", "n", "left", "rate", "ci_low", "ci_high", "lift", "share"])
    overall = df[TARGET].mean()
    g = df.groupby(col, observed=True)[TARGET].agg(["size", "sum"]).reset_index()
    g.columns = ["group", "n", "left"]
    g = g[g["n"] >= min_n].copy()
    ci = [wilson_interval(int(l), int(n)) for l, n in zip(g["left"], g["n"])]
    g["rate"] = g["left"] / g["n"]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["lift"] = g["rate"] / overall if overall else np.nan
    g["share"] = g["n"] / len(df)
    g["group"] = g["group"].astype(str)
    if sort == "rate":
        g = g.sort_values("rate", ascending=False)
    return g.reset_index(drop=True)


def crosstab_rates(df: pd.DataFrame, row: str, col: str, min_n: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way attrition-rate table and the matching counts (cells under ``min_n`` set to NaN)."""
    rates = df.pivot_table(index=row, columns=col, values=TARGET, aggfunc="mean", observed=True)
    counts = df.pivot_table(index=row, columns=col, values=TARGET, aggfunc="size", observed=True)
    rates = rates.where(counts >= min_n)
    return rates, counts


def _cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    if n == 0 or min(r, k) < 2:
        return float("nan")
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def _bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def association_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank every raw feature by strength of association with attrition.

    Categorical and ordinal features: chi-square test with Cramer's V.
    Continuous features: Mann-Whitney U test with rank-biserial correlation
    (sign > 0 means leavers tend to have *higher* values).
    p-values are adjusted with Benjamini-Hochberg across all features.
    """
    rows = []
    y = df[TARGET].to_numpy()
    for col in schema.MODEL_FEATURES:
        s = df[col]
        if col in schema.NOMINAL_FEATURES or col in schema.ORDINAL_FEATURES:
            table = pd.crosstab(s, y).to_numpy()
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
            rows.append({"feature": col, "label": schema.PRETTY_NAMES.get(col, col), "test": "chi-square",
                         "effect": _cramers_v(table), "effect_name": "Cramer's V", "direction": "",
                         "statistic": chi2, "p_value": p})
        else:
            a, b = s[y == 1].to_numpy(), s[y == 0].to_numpy()
            if len(a) == 0 or len(b) == 0:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rbc = 2 * u / (len(a) * len(b)) - 1
            rows.append({"feature": col, "label": schema.PRETTY_NAMES.get(col, col), "test": "Mann-Whitney U",
                         "effect": abs(rbc), "effect_name": "|rank-biserial r|",
                         "direction": "higher among leavers" if rbc > 0 else "lower among leavers",
                         "statistic": u, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adjusted"] = _bh(out["p_value"].to_numpy())
    out["significant"] = out["p_adjusted"] < 0.05
    return out.sort_values("effect", ascending=False).reset_index(drop=True)


def numeric_comparison(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or schema.NUMERIC_FEATURES
    rows = []
    for c in cols:
        a, b = df.loc[df[TARGET] == 1, c], df.loc[df[TARGET] == 0, c]
        if len(a) == 0 or len(b) == 0:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": c, "label": schema.PRETTY_NAMES.get(c, c),
                     "median_left": float(a.median()), "median_stayed": float(b.median()),
                     "mean_left": float(a.mean()), "mean_stayed": float(b.mean()),
                     "rank_biserial": 2 * u / (len(a) * len(b)) - 1, "p_value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adjusted"] = _bh(out["p_value"].to_numpy())
    return out


def headline_findings(df: pd.DataFrame, k: int = 5, min_n: int = 30) -> list[dict]:
    """The k largest group-level gaps in attrition, written from the numbers.

    For each categorical dimension the highest-rate group with at least
    ``min_n`` employees is compared with the rest of the (filtered) workforce.
    Only gaps whose Wilson intervals do not overlap are reported.
    """
    if len(df) < min_n * 2:
        return []
    found = []
    for key, (col, label, _) in DIMENSIONS.items():
        if col not in df.columns:
            continue
        r = rate_by(df, col, min_n=min_n)
        if len(r) < 2:
            continue
        top = r.sort_values("rate", ascending=False).iloc[0]
        rest = df[df[col].astype(str) != top["group"]]
        if len(rest) < min_n:
            continue
        rest_left = int(rest[TARGET].sum())
        rest_rate = rest_left / len(rest)
        rlo, rhi = wilson_interval(rest_left, len(rest))
        if top["ci_low"] <= rhi:
            continue
        found.append({
            "dimension": label, "column": col, "group": top["group"], "n": int(top["n"]),
            "rate": float(top["rate"]), "ci": (float(top["ci_low"]), float(top["ci_high"])),
            "rest_rate": rest_rate, "rest_ci": (rlo, rhi), "ratio": float(top["rate"] / rest_rate) if rest_rate else np.nan,
            "text": (f"{label} = {top['group']}: {top['rate']:.0%} attrition (95% CI {top['ci_low']:.0%}-{top['ci_high']:.0%}, "
                     f"n = {int(top['n'])}) against {rest_rate:.0%} for everyone else."),
        })
    found.sort(key=lambda d: d["ratio"], reverse=True)
    # keep one finding per underlying concept (e.g. job level and income band move together)
    return found[:k]


def correlation_matrix(df: pd.DataFrame, cols: list[str] | None = None, method: str = "spearman") -> pd.DataFrame:
    cols = cols or (schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES + [TARGET])
    return df[cols].corr(method=method)


def cramers_v_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or (schema.NOMINAL_FEATURES + [schema.TARGET])
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = _cramers_v(pd.crosstab(df[a], df[b]).to_numpy())
            m.loc[a, b] = m.loc[b, a] = v
    return m


def outlier_table(df: pd.DataFrame, cols: list[str] | None = None, k: float = 1.5) -> pd.DataFrame:
    cols = cols or schema.NUMERIC_FEATURES
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"feature": c, "label": schema.PRETTY_NAMES.get(c, c), "q1": q1, "median": s.median(), "q3": q3,
                     "lower_fence": lo, "upper_fence": hi, "outliers": n_out, "outlier_pct": n_out / max(len(s), 1),
                     "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows).sort_values("outliers", ascending=False).reset_index(drop=True)
