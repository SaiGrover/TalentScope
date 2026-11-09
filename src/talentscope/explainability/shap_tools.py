"""SHAP explanations for the selected pipeline.

The pipeline's imputer and encoder are applied first (never the sampler), and
the explainer is built for the final estimator on that transformed matrix, so
the feature names line up with the one-hot columns the model actually saw.

Explainer per model family:

* tree models (decision tree, random forest, XGBoost, LightGBM) -> ``shap.TreeExplainer``
* logistic regression -> ``shap.LinearExplainer`` with a training-set background
* SVM -> ``shap.KernelExplainer`` on the positive-class probability, with a
  k-means summary of the training set as background

SHAP values describe how the *model* moves a prediction away from its average
output. They are associations learned from synthetic data, not causes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from talentscope import config
from talentscope.data import schema
from talentscope.preprocessing import NUMERIC_BLOCK, NOMINAL_BLOCK, feature_names, transform_features

OUTPUT_SPACE = {
    "logreg": "log-odds", "xgboost": "log-odds", "lightgbm": "log-odds",
    "tree": "probability", "forest": "probability", "svm": "probability", "dummy": "probability",
}


def output_space(model_key: str) -> str:
    return OUTPUT_SPACE.get(model_key, "probability")


@dataclass
class ShapExplanation:
    values: np.ndarray             # (n_rows, n_features) contributions for the "left" class
    base_value: float
    feature_names: list[str]       # transformed names, e.g. cat__OverTime_Yes
    display_data: np.ndarray       # unscaled feature values for colouring / dependence plots
    row_ids: np.ndarray
    output_space: str
    explainer_kind: str

    @property
    def pretty_names(self) -> list[str]:
        return [schema.pretty_feature(n) for n in self.feature_names]

    def importance(self) -> pd.DataFrame:
        imp = np.abs(self.values).mean(axis=0)
        return (pd.DataFrame({"feature": self.feature_names, "label": self.pretty_names,
                              "source": [schema.source_feature(n) for n in self.feature_names],
                              "mean_abs_shap": imp})
                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def display_matrix(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Imputed numeric values (not scaled) followed by the one-hot block."""
    Xi = pipeline.named_steps["impute"].transform(X[schema.MODEL_FEATURES])
    cat = pipeline.named_steps["encode"].named_transformers_["cat"].transform(Xi[NOMINAL_BLOCK])
    return np.hstack([Xi[NUMERIC_BLOCK].to_numpy(dtype=float), np.asarray(cat, dtype=float)])


def build_explainer(pipeline, model_key: str, X_background: pd.DataFrame, seed: int = config.RANDOM_SEED):
    import shap

    est = pipeline.named_steps["model"]
    kind = next((spec_kind for k, spec_kind in [("logreg", "linear"), ("svm", "kernel"), ("dummy", "none")]
                 if k == model_key), "tree")
    if kind == "none":
        raise ValueError("The dummy baseline predicts a constant; there is nothing to explain.")
    bg = transform_features(pipeline, X_background)
    if kind == "tree":
        explainer = shap.TreeExplainer(est)
    elif kind == "linear":
        explainer = shap.LinearExplainer(est, shap.maskers.Independent(bg, max_samples=len(bg)))
    else:
        rng = np.random.default_rng(seed)
        summary = shap.kmeans(bg, 20)
        f = lambda Z: est.predict_proba(Z)[:, 1]  # noqa: E731
        explainer = shap.KernelExplainer(f, summary)
        explainer._ts_seed = int(rng.integers(0, 2 ** 31))
    explainer._ts_kind = kind
    return explainer


def _positive_class(values, base):
    values = np.asarray(values)
    base = np.asarray(base)
    if values.ndim == 3:            # (rows, features, classes)
        values = values[:, :, 1]
        base = base[..., 1] if base.ndim >= 1 and base.shape[-1] == 2 else base
    if base.ndim > 0:
        base = base.reshape(-1)[0] if base.size == 1 else base.reshape(-1)[-1] if base.size == 2 else base.mean()
    return values, float(base)


def explain(pipeline, explainer, X: pd.DataFrame, model_key: str, kernel_nsamples: int = 300) -> ShapExplanation:
    Z = transform_features(pipeline, X)
    kind = getattr(explainer, "_ts_kind", "tree")
    if kind == "kernel":
        np.random.seed(getattr(explainer, "_ts_seed", config.RANDOM_SEED))
        vals = explainer.shap_values(Z, nsamples=kernel_nsamples, silent=True)
        base = explainer.expected_value
    elif kind == "tree":
        out = explainer(Z, check_additivity=False)
        vals, base = out.values, out.base_values
    else:
        out = explainer(Z)
        vals, base = out.values, out.base_values
    vals, base = _positive_class(vals, base)
    return ShapExplanation(values=np.asarray(vals, dtype=float), base_value=base,
                           feature_names=feature_names(pipeline), display_data=display_matrix(pipeline, X),
                           row_ids=np.asarray(X.index), output_space=output_space(model_key), explainer_kind=kind)


def aggregate_by_source(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Sum one-hot contributions back to their raw column (rows x 30 features)."""
    src = [schema.source_feature(n) for n in names]
    df = pd.DataFrame(values, columns=names)
    return df.T.groupby(src, sort=False).sum().T


def permutation_table(pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 20,
                      seed: int = config.RANDOM_SEED, n_jobs: int = -1) -> pd.DataFrame:
    """Drop in PR-AUC when one raw feature is shuffled (on the held-out set)."""
    from sklearn.inspection import permutation_importance

    res = permutation_importance(pipeline, X[schema.MODEL_FEATURES], y, scoring="average_precision",
                                 n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs)
    return (pd.DataFrame({"feature": schema.MODEL_FEATURES,
                          "label": [schema.PRETTY_NAMES.get(f, f) for f in schema.MODEL_FEATURES],
                          "importance_mean": res.importances_mean, "importance_std": res.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def explain_bundle(bundle, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series,
                   groups: pd.DataFrame | None = None, kernel_nsamples: int = 300, n_repeats: int = 20) -> Path:
    """Compute and save SHAP values, permutation importance and group metrics for a run."""
    from talentscope.evaluation import group_metrics
    from talentscope.modeling.artifacts import write_json

    out = bundle.explain_dir
    out.mkdir(parents=True, exist_ok=True)
    pipeline = bundle.load_model()
    key = bundle.selection["model"]
    meta = {"model": key, "output_space": output_space(key)}
    if key != "dummy":
        explainer = build_explainer(pipeline, key, X_train)
        exp = explain(pipeline, explainer, X_test, key, kernel_nsamples)
        np.savez_compressed(out / "shap_test.npz", values=exp.values, display=exp.display_data,
                            row_ids=exp.row_ids, base_value=np.array([exp.base_value]),
                            feature_names=np.array(exp.feature_names))
        exp.importance().to_csv(out / "shap_importance.csv", index=False)
        agg = aggregate_by_source(exp.values, exp.feature_names)
        (agg.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").rename_axis("feature")
         .reset_index().to_csv(out / "shap_importance_by_feature.csv", index=False))
        meta.update({"explainer": exp.explainer_kind, "rows_explained": int(len(X_test)),
                     "base_value": exp.base_value, "explained_set": "held-out test set",
                     "kernel_nsamples": kernel_nsamples if exp.explainer_kind == "kernel" else None})
    perm = permutation_table(pipeline, X_test, y_test, n_repeats=n_repeats)
    perm.to_csv(out / "permutation_importance.csv", index=False)
    meta["permutation"] = {"set": "held-out test set", "scoring": "average_precision", "n_repeats": n_repeats}
    if groups is not None:
        from talentscope.evaluation import group_metrics as gm
        proba = pipeline.predict_proba(X_test[schema.MODEL_FEATURES])[:, 1]
        frames = []
        for col in groups.columns:
            g = gm(y_test, proba, bundle.threshold, groups[col].reindex(X_test.index), min_size=10)
            g.insert(0, "attribute", col)
            frames.append(g)
        pd.concat(frames, ignore_index=True).to_csv(out / "group_metrics.csv", index=False)
    write_json(out / "explain_metadata.json", meta)
    return out


def load_explanations(bundle) -> dict:
    d = bundle.explain_dir
    out = {}
    if (d / "shap_test.npz").exists():
        z = np.load(d / "shap_test.npz", allow_pickle=False)
        out["shap"] = ShapExplanation(values=z["values"], base_value=float(z["base_value"][0]),
                                      feature_names=[str(s) for s in z["feature_names"]],
                                      display_data=z["display"], row_ids=z["row_ids"],
                                      output_space=output_space(bundle.selection["model"]),
                                      explainer_kind=bundle.selection.get("explainer", ""))
    for name in ("shap_importance", "shap_importance_by_feature", "permutation_importance", "group_metrics"):
        p = d / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    if (d / "explain_metadata.json").exists():
        import json
        out["meta"] = json.loads((d / "explain_metadata.json").read_text(encoding="utf-8"))
    return out
