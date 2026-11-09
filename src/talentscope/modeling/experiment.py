"""The experiment runner behind the Model Laboratory, the notebooks and the default model.

Protocol (every step is in this file, nothing is hand-edited afterwards):

1. **Split once.** Stratified 80/20 split of the 1,470 employees with a fixed
   seed. The 20% test set is not touched again until step 6.
2. **Tune.** For every (model, imbalance strategy) pair, ``RandomizedSearchCV``
   with stratified K-fold on the *training set only*, scoring PR-AUC. The
   whole pipeline (imputer, sampler, scaler, encoder, model) is refitted in
   every fold, so SMOTENC never sees a validation row.
3. **Compare.** The tuned configuration is re-evaluated with *repeated*
   stratified K-fold using a different seed from the search. Fold-level
   threshold-free metrics give a mean and standard deviation; out-of-fold
   probabilities (averaged over repeats) give the pooled curves.
4. **Select a model.** Highest mean CV PR-AUC, or, with the one-standard-error
   rule (default), the simplest configuration whose mean PR-AUC is within one
   standard error of the best. Uses training data only.
5. **Select a threshold.** The threshold that maximises F1 (or F2) on the
   selected model's out-of-fold training predictions.
6. **Test once.** The selected pipeline is refitted on the full training set
   and scored once on the untouched test set, with stratified bootstrap
   confidence intervals.

The remaining configurations are never scored on the test set, so the test
estimate is not inflated by picking the luckiest model.
"""
from __future__ import annotations

import json
import platform
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from talentscope import __version__, config
from talentscope.data import dataset_card, load_dataset, schema, split_features_target
from talentscope.evaluation import (
    bootstrap_ci,
    calibration_table,
    classification_report_frame,
    compute_metrics,
    curve_points,
    select_threshold,
)
from talentscope.modeling.registry import MODEL_SPECS, STRATEGIES, applicable_strategies, build_pipeline
from talentscope.preprocessing import make_split

# Simplest first. Used by the one-standard-error rule to break near-ties.
COMPLEXITY = {"dummy": 0, "logreg": 1, "tree": 2, "svm": 3, "forest": 4, "lightgbm": 5, "xgboost": 6}
STRATEGY_COMPLEXITY = {"none": 0, "class_weight": 1, "random_oversample": 2, "smotenc": 3}
FOLD_METRICS = ["pr_auc", "roc_auc", "brier", "f1", "precision", "recall", "balanced_accuracy", "accuracy"]


@dataclass
class ExperimentConfig:
    models: list[str] = field(default_factory=lambda: list(MODEL_SPECS))
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES))
    n_iter: int = 30
    cv_folds: int = config.CV_FOLDS
    cv_repeats: int = 3
    seed: int = config.RANDOM_SEED
    test_size: float = config.TEST_SIZE
    smote_sampling_strategy: float = 1.0
    smote_k_neighbors: int = 5
    threshold_metric: str = "f1"          # "f1" or "f2"
    selection_rule: str = "one_se"        # "one_se" or "best"
    bootstrap_resamples: int = config.BOOTSTRAP_RESAMPLES
    n_jobs: int = -1
    name: str = "experiment"

    def combos(self) -> list[tuple[str, str]]:
        out = []
        for m in self.models:
            for s in applicable_strategies(m, self.strategies):
                out.append((m, s))
        return out


@dataclass
class ExperimentResult:
    config: dict
    metadata: dict
    cv_summary: pd.DataFrame
    cv_folds: pd.DataFrame
    search_trials: pd.DataFrame
    best_params: dict
    oof: pd.DataFrame
    selection: dict
    test_predictions: pd.DataFrame
    test_metrics: dict
    test_bootstrap: pd.DataFrame
    test_curves: dict
    test_calibration: pd.DataFrame
    oof_calibration: pd.DataFrame
    classification_report: pd.DataFrame
    resampling: dict
    model: object


def combo_id(model: str, strategy: str) -> str:
    return f"{model}|{strategy}"


def _clean_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        k = k.replace("model__", "")
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        out[k] = v
    return out


def resampling_evidence(X_train, y_train, cfg: ExperimentConfig) -> dict:
    """Class counts before and after each sampler on the first training fold.

    This is the evidence that resampling happens inside a training fold: the
    validation fold and the test set keep their original counts.
    """
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    tr_idx, va_idx = next(skf.split(X_train, y_train))
    Xf, yf = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    evidence = {
        "train_set": {"stayed": int((y_train == 0).sum()), "left": int((y_train == 1).sum())},
        "fold_train_original": {"stayed": int((yf == 0).sum()), "left": int((yf == 1).sum())},
        "fold_validation": {"stayed": int((y_train.iloc[va_idx] == 0).sum()),
                            "left": int((y_train.iloc[va_idx] == 1).sum())},
        "after": {},
    }
    for strategy in ("smotenc", "random_oversample"):
        if strategy not in cfg.strategies:
            continue
        pipe = build_pipeline("logreg", strategy, cfg.seed, yf,
                              smote_sampling_strategy=cfg.smote_sampling_strategy,
                              smote_k_neighbors=cfg.smote_k_neighbors)
        Xi = pipe.named_steps["impute"].fit_transform(Xf)
        _, yr = pipe.named_steps["sampler"].fit_resample(Xi, yf)
        yr = np.asarray(yr)
        evidence["after"][strategy] = {"stayed": int((yr == 0).sum()), "left": int((yr == 1).sum())}
    return evidence


def run_experiment(cfg: ExperimentConfig, progress: Callable[[float, str], None] | None = None,
                   data: pd.DataFrame | None = None) -> ExperimentResult:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()
    progress = progress or (lambda f, m: None)
    df = load_dataset() if data is None else data
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_split(X, y, cfg.test_size, cfg.seed)

    combos = cfg.combos()
    if not combos:
        raise ValueError("No (model, strategy) combination selected.")
    search_cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    # a different seed for evaluation folds, so the comparison is not scored on the
    # exact folds the search optimised against
    eval_cv = RepeatedStratifiedKFold(n_splits=cfg.cv_folds, n_repeats=cfg.cv_repeats, random_state=cfg.seed + 1)
    splits = list(eval_cv.split(X_train, y_train))

    trials, folds, summaries, best_params, oof_cols = [], [], [], {}, {}
    tuned = {}
    for i, (m, s) in enumerate(combos):
        cid = combo_id(m, s)
        spec = MODEL_SPECS[m]
        progress(i / (len(combos) + 2), f"Tuning {spec.name} with {STRATEGIES[s].name.lower()} ({i + 1}/{len(combos)})")
        pipe = build_pipeline(m, s, cfg.seed, y_train, smote_sampling_strategy=cfg.smote_sampling_strategy,
                              smote_k_neighbors=cfg.smote_k_neighbors)
        t_combo = time.time()
        if spec.search_space and cfg.n_iter > 0:
            search = RandomizedSearchCV(pipe, spec.search_space, n_iter=cfg.n_iter, scoring="average_precision",
                                        cv=search_cv, random_state=cfg.seed, n_jobs=cfg.n_jobs, refit=False,
                                        error_score=np.nan)
            search.fit(X_train, y_train)
            res = pd.DataFrame(search.cv_results_)
            for _, r in res.iterrows():
                trials.append({"combo": cid, "model": m, "strategy": s,
                               "params": json.dumps(_clean_params(r["params"]), default=str),
                               "mean_cv_pr_auc": r["mean_test_score"], "std_cv_pr_auc": r["std_test_score"],
                               "rank": int(r["rank_test_score"])})
            params = search.best_params_
            pipe.set_params(**params)
        else:
            params = {}
        best_params[cid] = _clean_params(params)
        tuned[cid] = pipe

        # repeated CV evaluation of the tuned configuration
        oof_sum = np.zeros(len(y_train))
        oof_cnt = np.zeros(len(y_train))
        for k, (tr, va) in enumerate(splits):
            est = clone(pipe)
            est.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = est.predict_proba(X_train.iloc[va])[:, 1]
            oof_sum[va] += p
            oof_cnt[va] += 1
            met = compute_metrics(y_train.iloc[va], p, 0.5)
            folds.append({"combo": cid, "model": m, "strategy": s, "fold": k,
                          **{f: met[f] for f in FOLD_METRICS}})
        oof = oof_sum / np.maximum(oof_cnt, 1)
        oof_cols[cid] = oof
        thr, thr_score = select_threshold(y_train, oof, cfg.threshold_metric)
        pooled = compute_metrics(y_train, oof, thr)
        fdf = pd.DataFrame([f for f in folds if f["combo"] == cid])
        row = {"combo": cid, "model": m, "model_name": spec.name, "strategy": s,
               "strategy_name": STRATEGIES[s].name, "n_folds": len(fdf),
               "fit_seconds": round(time.time() - t_combo, 1), "oof_threshold": thr}
        for f in FOLD_METRICS:
            row[f"cv_{f}_mean"] = float(fdf[f].mean())
            row[f"cv_{f}_std"] = float(fdf[f].std(ddof=1))
        for f in ["pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "brier"]:
            row[f"oof_{f}"] = pooled[f]
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary["cv_pr_auc_se"] = summary["cv_pr_auc_std"] / np.sqrt(summary["n_folds"])
    summary["complexity"] = summary["model"].map(COMPLEXITY).fillna(9) * 10 + summary["strategy"].map(STRATEGY_COMPLEXITY)
    best_row = summary.loc[summary["cv_pr_auc_mean"].idxmax()]
    if cfg.selection_rule == "one_se":
        floor = best_row["cv_pr_auc_mean"] - best_row["cv_pr_auc_se"]
        eligible = summary[(summary["cv_pr_auc_mean"] >= floor) & (summary["model"] != "dummy")]
        chosen = eligible.sort_values(["complexity", "cv_pr_auc_mean"], ascending=[True, False]).iloc[0]
        reason = (f"One-standard-error rule: the best mean CV PR-AUC was {best_row['cv_pr_auc_mean']:.3f} "
                  f"({best_row['model_name']}, {best_row['strategy_name']}); the simplest configuration within one "
                  f"standard error ({best_row['cv_pr_auc_se']:.3f}) of it was selected.")
    else:
        chosen = best_row
        floor = None
        reason = "Highest mean cross-validated PR-AUC on the training set."
    summary["selected"] = summary["combo"] == chosen["combo"]
    summary = summary.sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)

    cid = chosen["combo"]
    m, s = cid.split("|")
    threshold = float(chosen["oof_threshold"])
    progress((len(combos)) / (len(combos) + 2), "Refitting the selected pipeline on the full training set")
    final = clone(tuned[cid])
    final.fit(X_train, y_train)

    progress((len(combos) + 1) / (len(combos) + 2), "Scoring the untouched test set once")
    test_proba = final.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)
    test_metrics_05 = compute_metrics(y_test, test_proba, 0.5)
    boot = bootstrap_ci(y_test, test_proba, threshold, cfg.bootstrap_resamples, config.CONFIDENCE_LEVEL, cfg.seed)

    raw_ids = load_ids(df)
    test_pred = pd.DataFrame({
        "row_id": X_test.index, "EmployeeNumber": raw_ids.reindex(X_test.index).to_numpy(),
        "actual_left": y_test.to_numpy(), "probability": test_proba,
        "predicted_left": (test_proba >= threshold).astype(int),
    })
    oof_df = pd.DataFrame(oof_cols, index=X_train.index)
    oof_df.insert(0, "actual_left", y_train.to_numpy())
    oof_df.index.name = "row_id"

    selection = {
        "combo": cid, "model": m, "model_name": MODEL_SPECS[m].name, "strategy": s,
        "strategy_name": STRATEGIES[s].name, "rule": cfg.selection_rule, "reason": reason,
        "best_combo_by_mean": best_row["combo"], "one_se_floor": floor,
        "cv_pr_auc_mean": float(chosen["cv_pr_auc_mean"]), "cv_pr_auc_std": float(chosen["cv_pr_auc_std"]),
        "threshold": threshold, "threshold_metric": cfg.threshold_metric,
        "threshold_source": "out-of-fold predictions on the training set (repeated stratified CV)",
        "params": best_params[cid], "explainer": MODEL_SPECS[m].explainer,
    }
    import sklearn, imblearn
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                "imbalanced-learn": imblearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    for mod in ("xgboost", "lightgbm", "shap"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    metadata = {
        "talentscope_version": __version__,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(time.time() - t0, 1),
        "dataset": dataset_card(),
        "split": {"train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
                  "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum()),
                  "stratified": True, "seed": cfg.seed, "test_size": cfg.test_size},
        "protocol": {
            "search": f"RandomizedSearchCV, {cfg.n_iter} candidates, {cfg.cv_folds}-fold stratified CV (seed {cfg.seed}), scoring PR-AUC",
            "comparison": f"{cfg.cv_repeats} x {cfg.cv_folds}-fold repeated stratified CV (seed {cfg.seed + 1})",
            "threshold": f"maximise {cfg.threshold_metric.upper()} on out-of-fold training predictions",
            "test": "single evaluation of the selected pipeline; stratified percentile bootstrap "
                    f"({cfg.bootstrap_resamples} resamples) for 95% intervals",
            "resampling": "inside the imblearn pipeline, so only training folds are resampled",
        },
        "versions": versions,
        "features": schema.MODEL_FEATURES,
        "test_metrics_at_0_5": test_metrics_05,
    }
    progress(1.0, "Done")
    return ExperimentResult(
        config=asdict(cfg), metadata=metadata, cv_summary=summary, cv_folds=pd.DataFrame(folds),
        search_trials=pd.DataFrame(trials), best_params=best_params, oof=oof_df, selection=selection,
        test_predictions=test_pred, test_metrics=test_metrics, test_bootstrap=boot,
        test_curves=curve_points(y_test, test_proba), test_calibration=calibration_table(y_test, test_proba, 5),
        oof_calibration=calibration_table(y_train, oof_df[cid], 10),
        classification_report=classification_report_frame(y_test, test_proba, threshold),
        resampling=resampling_evidence(X_train, y_train, cfg), model=final,
    )


def load_ids(df: pd.DataFrame) -> pd.Series:
    """EmployeeNumber per row of the prepared frame (read from the raw file, never a model input)."""
    from talentscope.data import load_raw
    raw = load_raw()
    return raw[schema.ID_COLUMNS[0]].reindex(df.index)
