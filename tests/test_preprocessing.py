"""Split, preprocessing, imputation and the guarantee that resampling stays inside training folds."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from talentscope.data import schema
from talentscope.modeling import build_pipeline
from talentscope.preprocessing import feature_names, transform_features


def test_split_is_stratified_and_disjoint(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 1176 and len(X_test) == 294
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert int(y_test.sum()) == 47


def test_split_is_reproducible(df):
    from talentscope.data import split_features_target
    from talentscope.preprocessing import make_split
    X, y = split_features_target(df)
    a = make_split(X, y)[1].index
    b = make_split(X, y)[1].index
    assert list(a) == list(b)


def test_transformed_matrix_shape_and_names(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("logreg", "none", 42, y_train).fit(X_train, y_train)
    Z = transform_features(pipe, X_test)
    names = feature_names(pipe)
    assert Z.shape == (len(X_test), len(names))
    n_onehot = sum(X_train[c].nunique() for c in schema.NOMINAL_FEATURES)
    assert len(names) == len(schema.NUMERIC_FEATURES) + len(schema.ORDINAL_FEATURES) + n_onehot
    assert np.isfinite(Z).all()


def test_scaler_is_fitted_on_training_rows_only(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("logreg", "none", 42, y_train).fit(X_train, y_train)
    scaler = pipe.named_steps["encode"].named_transformers_["num"].named_steps["scale"]
    col = (schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES).index("MonthlyIncome")
    assert scaler.mean_[col] == pytest.approx(X_train["MonthlyIncome"].mean())
    assert scaler.mean_[col] != pytest.approx(pd.concat([X_train, X_test])["MonthlyIncome"].mean())


def test_missing_values_are_imputed_with_training_statistics(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("logreg", "none", 42, y_train).fit(X_train, y_train)
    holes = X_test.head(5).copy()
    holes.loc[holes.index[0], "MonthlyIncome"] = np.nan
    holes.loc[holes.index[1], "JobRole"] = np.nan
    holes.loc[holes.index[2], "OverTime"] = np.nan
    proba = pipe.predict_proba(holes)[:, 1]
    assert np.isfinite(proba).all()
    imputed = pipe.named_steps["impute"].transform(holes)
    assert imputed.loc[holes.index[0], "MonthlyIncome"] == X_train["MonthlyIncome"].median()
    assert imputed.loc[holes.index[1], "JobRole"] == X_train["JobRole"].mode().iloc[0]


def test_unknown_category_does_not_break_prediction(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("logreg", "none", 42, y_train).fit(X_train, y_train)
    odd = X_test.head(1).copy()
    odd["JobRole"] = "Chief Vibes Officer"
    assert 0 <= pipe.predict_proba(odd)[0, 1] <= 1


def test_smotenc_outputs_valid_rows(split):
    X_train, _, y_train, _ = split
    pipe = build_pipeline("logreg", "smotenc", 42, y_train)
    Xi = pipe.named_steps["impute"].fit_transform(X_train)
    Xr, yr = pipe.named_steps["sampler"].fit_resample(Xi, y_train)
    assert (np.asarray(yr) == 1).sum() == (np.asarray(yr) == 0).sum()
    for c in schema.NOMINAL_FEATURES:
        assert set(Xr[c].unique()) <= set(X_train[c].unique())
    Z = pipe.named_steps["encode"].fit_transform(Xr)
    ordinal_cols = [(schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES).index(c) for c in schema.ORDINAL_FEATURES]
    raw_num = pipe.named_steps["encode"].named_transformers_["num"].named_steps["round"].transform(
        Xr[schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES])
    assert np.allclose(raw_num[:, ordinal_cols], np.round(raw_num[:, ordinal_cols]))


class _RecordingSMOTENC(SMOTENC):
    """SMOTENC that records the size of every set it resamples."""
    calls: list = []

    def fit_resample(self, X, y, **kw):
        _RecordingSMOTENC.calls.append(len(y))
        return super().fit_resample(X, y, **kw)


def test_smote_runs_only_on_training_folds(split):
    X_train, _, y_train, _ = split
    pipe = build_pipeline("logreg", "smotenc", 42, y_train)
    pipe.steps[1] = ("sampler", _RecordingSMOTENC(categorical_features=list(schema.NOMINAL_FEATURES), random_state=0))
    _RecordingSMOTENC.calls = []
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
    fold_train_sizes = sorted(len(tr) for tr, _ in cv.split(X_train, y_train))
    assert sorted(_RecordingSMOTENC.calls) == fold_train_sizes  # never the validation fold, never the full set
    _RecordingSMOTENC.calls = []
    pipe.fit(X_train, y_train)
    pipe.predict_proba(X_train.head(50))
    assert _RecordingSMOTENC.calls == [len(y_train)]  # predict does not resample
