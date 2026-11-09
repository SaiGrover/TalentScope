"""Dataset schema, validation and derived bands."""
import numpy as np
import pandas as pd

from talentscope.data import dataset_card, prepare_dataset, schema, validate_dataset


def test_shape_and_target(raw):
    assert raw.shape == (schema.EXPECTED_ROWS, schema.EXPECTED_COLUMNS)
    assert set(raw["Attrition"].unique()) == {"Yes", "No"}
    assert int((raw["Attrition"] == "Yes").sum()) == schema.EXPECTED_POSITIVES


def test_attrition_rate_is_about_16_percent(raw):
    rate = (raw["Attrition"] == "Yes").mean()
    assert abs(rate - 0.1612) < 0.001


def test_schema_columns_documented(raw):
    assert set(raw.columns) == set(schema.ALL_COLUMNS)
    assert len(schema.MODEL_FEATURES) == 30
    assert not set(schema.DROP_COLUMNS) & set(schema.MODEL_FEATURES)


def test_validation_passes_on_real_file(raw):
    rep = validate_dataset(raw)
    assert rep.passed, rep.to_frame()
    assert set(rep.constant_columns) == {"EmployeeCount", "Over18", "StandardHours"}
    assert rep.duplicates == 0
    assert rep.leakage_screen["univariate_auc"].max() < 0.9


def test_validation_catches_problems(raw):
    bad = raw.copy()
    bad.loc[0, "Attrition"] = "Maybe"
    bad.loc[1, "Age"] = -3
    bad.loc[2, "MonthlyIncome"] = np.nan
    bad = pd.concat([bad, bad.iloc[[5]]], ignore_index=True)
    rep = validate_dataset(bad)
    status = dict(zip(rep.to_frame()["name"], rep.to_frame()["status"]))
    assert status["Target labels"] == "fail"
    assert status["Missing values"] == "warn"
    assert status["Duplicate rows"] == "warn"
    assert status["Numeric values within plausible ranges"] == "warn"
    assert status["EmployeeNumber is unique"] == "fail"


def test_leakage_screen_flags_a_label_proxy(raw):
    leaky = raw.copy()
    leaky["DistanceFromHome"] = np.where(leaky["Attrition"] == "Yes", 29, 1)
    rep = validate_dataset(leaky)
    row = rep.to_frame().set_index("name").loc["Single-feature leakage screen"]
    assert row["status"] == "fail" and "DistanceFromHome" in row["detail"]


def test_prepare_drops_identifier_and_constants(raw):
    out = prepare_dataset(raw)
    for c in schema.DROP_COLUMNS:
        assert c not in out.columns
    assert out[schema.TARGET_FLAG].sum() == schema.EXPECTED_POSITIVES


def test_bands_cover_every_row(df):
    for c in ("AgeBand", "IncomeBand", "TenureBand", "DistanceBand", "PromotionRecency", "JobSatisfactionLabel"):
        assert df[c].notna().all(), c


def test_dataset_card_is_synthetic_and_hashed(raw):
    card = dataset_card(raw)
    assert card["synthetic"] is True
    assert len(card["sha256"]) == 64
    assert card["rows"] == 1470
