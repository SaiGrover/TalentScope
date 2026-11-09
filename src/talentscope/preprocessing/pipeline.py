"""Feature preprocessing that is fitted inside every training fold.

Every model is an ``imblearn`` pipeline with the same four stages::

    impute  ->  [resample]  ->  encode  ->  model

* ``impute``   median for numeric/ordinal columns, most-frequent for nominal
               columns. Returns a DataFrame so the resampler can see which
               columns are categorical. The IBM data has no missing values;
               the step exists so the pipeline is safe on new records.
* ``resample`` optional. SMOTENC / random oversampling. ``imblearn`` runs a
               sampler only during ``fit``; ``predict`` and ``transform`` skip
               it, so validation folds and the test set are never resampled.
* ``encode``   integer-valued columns are rounded (SMOTENC interpolates, so a
               synthetic employee could otherwise have a job satisfaction of
               2.37), optionally standardised for scale-sensitive models, and
               nominal columns are one-hot encoded with unknown categories
               ignored.

Because all of this sits inside one estimator, ``cross_val_score`` and
``RandomizedSearchCV`` refit the imputer, the sampler, the scaler and the
encoder on the training part of each fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from talentscope import config
from talentscope.data import schema

NUMERIC_BLOCK = schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES
NOMINAL_BLOCK = schema.NOMINAL_FEATURES


def make_split(X: pd.DataFrame, y: pd.Series, test_size: float = config.TEST_SIZE,
               seed: int = config.RANDOM_SEED):
    """Stratified train/test split, done once, before anything is fitted."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def round_to_int(X):
    """Round integer-coded columns back to whole numbers (needed after SMOTENC)."""
    return np.rint(np.asarray(X, dtype=float))


def build_imputer() -> ColumnTransformer:
    imputer = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_BLOCK),
            ("cat", SimpleImputer(strategy="most_frequent"), NOMINAL_BLOCK),
        ],
        verbose_feature_names_out=False,
    )
    return imputer.set_output(transform="pandas")


def build_encoder(scale: bool) -> ColumnTransformer:
    num_steps = [("round", FunctionTransformer(round_to_int, feature_names_out="one-to-one"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", SkPipeline(num_steps), NUMERIC_BLOCK),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_BLOCK),
        ]
    )


def transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the fitted impute + encode stages (never the sampler) to raw rows."""
    Xi = pipeline.named_steps["impute"].transform(X[schema.MODEL_FEATURES])
    return pipeline.named_steps["encode"].transform(Xi)


def feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["encode"].get_feature_names_out())
