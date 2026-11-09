"""The models and class-imbalance strategies TalentScope compares.

Each model declares its estimator factory, whether it needs scaled inputs, how
it takes class weights and a deliberately small, meaningful search space.
Search spaces are prefixed with ``model__`` so they plug straight into the
pipeline built by :func:`build_pipeline`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from talentscope.data import schema
from talentscope.preprocessing import build_encoder, build_imputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LGBM = False


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str                      # "baseline" | "linear" | "tree" | "ensemble" | "kernel"
    factory: Callable[[int, float | None], object]   # (seed, scale_pos_weight or None) -> estimator
    scale: bool
    supports_class_weight: bool
    search_space: dict = field(default_factory=dict)
    explainer: str = "kernel"        # "tree" | "linear" | "kernel" | "none"


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    description: str


STRATEGIES: dict[str, Strategy] = {
    "none": Strategy("none", "No rebalancing",
                     "Train on the original 5.2 : 1 class ratio. The reference point every other strategy must beat."),
    "class_weight": Strategy("class_weight", "Class weights",
                             "Re-weight the loss so each class contributes equally (inverse-frequency weights); no rows are added."),
    "smotenc": Strategy("smotenc", "SMOTENC",
                        "Synthesise new minority rows inside each training fold by interpolating numeric features "
                        "between nearest minority neighbours and taking the majority value of nominal ones."),
    "random_oversample": Strategy("random_oversample", "Random oversampling",
                                  "Duplicate minority rows at random inside each training fold."),
}


def _dummy(seed, w):
    return DummyClassifier(strategy="prior")


def _logreg(seed, w):
    return LogisticRegression(max_iter=5000, random_state=seed, class_weight="balanced" if w else None)


def _tree(seed, w):
    return DecisionTreeClassifier(random_state=seed, class_weight="balanced" if w else None)


def _forest(seed, w):
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1,
                                  class_weight="balanced" if w else None)


def _svm(seed, w):
    return SVC(kernel="rbf", probability=True, random_state=seed, class_weight="balanced" if w else None)


def _xgb(seed, w):
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
                         colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                         random_state=seed, n_jobs=1, scale_pos_weight=w if w else 1.0, verbosity=0)


def _lgbm(seed, w):
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.9, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1,
                          class_weight="balanced" if w else None)


MODEL_SPECS: dict[str, ModelSpec] = {
    "dummy": ModelSpec("dummy", "Dummy (prior)", "baseline", _dummy, scale=False,
                       supports_class_weight=False, explainer="none"),
    "logreg": ModelSpec("logreg", "Logistic regression", "linear", _logreg, scale=True, supports_class_weight=True,
                        search_space={"model__C": loguniform(1e-3, 1e1)}, explainer="linear"),
    "tree": ModelSpec("tree", "Decision tree", "tree", _tree, scale=False, supports_class_weight=True,
                      search_space={"model__max_depth": randint(2, 9),
                                    "model__min_samples_leaf": randint(5, 60),
                                    "model__criterion": ["gini", "entropy"]},
                      explainer="tree"),
    "forest": ModelSpec("forest", "Random forest", "ensemble", _forest, scale=False, supports_class_weight=True,
                        search_space={"model__max_depth": [4, 6, 8, 12, None],
                                      "model__min_samples_leaf": randint(1, 20),
                                      "model__max_features": ["sqrt", 0.3, 0.5]},
                        explainer="tree"),
    "svm": ModelSpec("svm", "Support vector machine", "kernel", _svm, scale=True, supports_class_weight=True,
                     search_space={"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-4, 1e-1)},
                     explainer="kernel"),
}
if HAS_XGB:
    MODEL_SPECS["xgboost"] = ModelSpec(
        "xgboost", "XGBoost", "ensemble", _xgb, scale=False, supports_class_weight=True,
        search_space={"model__n_estimators": randint(100, 600), "model__max_depth": randint(2, 6),
                      "model__learning_rate": loguniform(0.01, 0.2), "model__subsample": uniform(0.6, 0.4),
                      "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 10),
                      "model__reg_lambda": loguniform(0.1, 10)},
        explainer="tree")
if HAS_LGBM:
    MODEL_SPECS["lightgbm"] = ModelSpec(
        "lightgbm", "LightGBM", "ensemble", _lgbm, scale=False, supports_class_weight=True,
        search_space={"model__n_estimators": randint(100, 600), "model__num_leaves": randint(4, 32),
                      "model__learning_rate": loguniform(0.01, 0.2), "model__min_child_samples": randint(5, 40),
                      "model__colsample_bytree": uniform(0.5, 0.5), "model__reg_lambda": loguniform(0.01, 10)},
        explainer="tree")


def applicable_strategies(model_key: str, strategies: list[str]) -> list[str]:
    """The dummy baseline ignores class balance, so it is only run once, without rebalancing."""
    spec = MODEL_SPECS[model_key]
    out = []
    for s in strategies:
        if spec.family == "baseline" and s != "none":
            continue
        if s == "class_weight" and not spec.supports_class_weight:
            continue
        out.append(s)
    return out


def build_pipeline(model_key: str, strategy: str, seed: int, y=None, *,
                   smote_sampling_strategy: float | str = 1.0, smote_k_neighbors: int = 5) -> Pipeline:
    """Assemble ``impute -> [resample] -> encode -> model`` for one (model, strategy) pair."""
    spec = MODEL_SPECS[model_key]
    weight = None
    if strategy == "class_weight":
        if y is None:
            weight = 1.0
        else:
            y = np.asarray(y)
            weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    steps = [("impute", build_imputer())]
    if strategy == "smotenc":
        steps.append(("sampler", SMOTENC(categorical_features=list(schema.NOMINAL_FEATURES),
                                         sampling_strategy=smote_sampling_strategy,
                                         k_neighbors=smote_k_neighbors, random_state=seed)))
    elif strategy == "random_oversample":
        steps.append(("sampler", RandomOverSampler(sampling_strategy=smote_sampling_strategy, random_state=seed)))
    steps.append(("encode", build_encoder(spec.scale)))
    steps.append(("model", spec.factory(seed, weight)))
    return Pipeline(steps)
