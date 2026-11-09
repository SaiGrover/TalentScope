"""Metric computation shared by the experiment runner, the notebooks and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "specificity": "Specificity",
    "brier": "Brier score",
}
THRESHOLD_FREE = ["pr_auc", "roc_auc", "brier"]


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "pr_auc": float(average_precision_score(y_true, proba)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def select_threshold(y_true, proba, metric: str = "f1", grid=None) -> tuple[float, float]:
    """Pick the decision threshold that maximises ``metric`` (f1 or f2).

    Called on out-of-fold training predictions only; the test set is never used
    to choose a threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    beta = 2.0 if metric == "f2" else 1.0
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def threshold_sweep(y_true, proba, grid=None) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "flagged_share": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def curve_points(y_true, proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": prec.tolist(), "recall": rec.tolist()}


def calibration_table(y_true, proba, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def bootstrap_ci(y_true, proba, threshold: float, n_resamples: int = 1000, level: float = 0.95,
                 seed: int = 42) -> pd.DataFrame:
    """Percentile bootstrap intervals for every metric on a fixed set of predictions.

    Resamples are stratified by class so every resample contains both classes.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    point = compute_metrics(y_true, proba, threshold)
    keys = list(METRIC_LABELS)
    draws = {k: [] for k in keys}
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        m = compute_metrics(y_true[idx], proba[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    a = (1 - level) / 2
    rows = []
    for k in keys:
        arr = np.asarray(draws[k], dtype=float)
        rows.append({"metric": k, "label": METRIC_LABELS[k], "estimate": point[k],
                     "ci_low": float(np.nanquantile(arr, a)), "ci_high": float(np.nanquantile(arr, 1 - a)),
                     "std": float(np.nanstd(arr))})
    return pd.DataFrame(rows)


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion (used for group attrition rates)."""
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def classification_report_frame(y_true, proba, threshold: float) -> pd.DataFrame:
    pred = (np.asarray(proba) >= threshold).astype(int)
    rep = classification_report(y_true, pred, target_names=["Stayed", "Left"], output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})


def group_metrics(y_true, proba, threshold: float, groups: pd.Series, min_size: int = 10) -> pd.DataFrame:
    """Per-group error rates for the responsible-AI review."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    groups = pd.Series(np.asarray(groups), name="group")
    rows = []
    for g in groups.dropna().unique():
        mask = (groups == g).to_numpy()
        n = int(mask.sum())
        if n < min_size:
            continue
        yt, yp, pp = y_true[mask], pred[mask], proba[mask]
        pos = int(yt.sum())
        neg = n - pos
        rows.append({
            "group": str(g), "n": n, "actual_leavers": pos,
            "observed_rate": yt.mean(), "mean_predicted": pp.mean(), "flagged_rate": yp.mean(),
            "recall_tpr": ((yp == 1) & (yt == 1)).sum() / pos if pos else np.nan,
            "false_positive_rate": ((yp == 1) & (yt == 0)).sum() / neg if neg else np.nan,
            "precision": ((yp == 1) & (yt == 1)).sum() / yp.sum() if yp.sum() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
