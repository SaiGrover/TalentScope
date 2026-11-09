"""Training, selection, threshold choice, serialisation and inference."""
import json

import joblib
import numpy as np
import pytest

from talentscope import config
from talentscope.data import schema
from talentscope.evaluation import compute_metrics
from talentscope.modeling import MODEL_SPECS, build_pipeline, load_bundle, save_bundle
from talentscope.modeling.inference import predict_employee, risk_band, typical_employee, validate_employee


@pytest.mark.parametrize("model", list(MODEL_SPECS))
def test_every_model_fits_and_predicts(model, split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline(model, "none", 42, y_train).fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    assert p.shape == (len(X_test),)
    assert ((p >= 0) & (p <= 1)).all()


def test_experiment_structure(tiny_result):
    r = tiny_result
    combos = set(r.cv_summary["combo"])
    assert combos == {"dummy|none", "logreg|none", "logreg|smotenc", "tree|none", "tree|smotenc"}
    assert r.cv_summary["selected"].sum() == 1
    assert r.selection["model"] != "dummy"
    assert set(r.cv_folds["combo"]) == combos
    assert len(r.oof) == 1176 and len(r.test_predictions) == 294


def test_threshold_comes_from_training_predictions(tiny_result):
    from talentscope.evaluation import select_threshold
    r = tiny_result
    y = r.oof["actual_left"].to_numpy()
    thr, _ = select_threshold(y, r.oof[r.selection["combo"]].to_numpy(), r.config["threshold_metric"])
    assert thr == pytest.approx(r.selection["threshold"])


def test_test_metrics_match_saved_predictions(tiny_result):
    r = tiny_result
    tp = r.test_predictions
    m = compute_metrics(tp["actual_left"], tp["probability"], r.selection["threshold"])
    for k in ("pr_auc", "roc_auc", "f1", "recall", "precision"):
        assert m[k] == pytest.approx(r.test_metrics[k])


def test_dummy_baseline_is_at_chance(tiny_result):
    d = tiny_result.cv_summary.set_index("combo").loc["dummy|none"]
    assert d["cv_roc_auc_mean"] == pytest.approx(0.5)
    assert abs(d["cv_pr_auc_mean"] - 0.1616) < 0.01


def test_resampling_evidence(tiny_result):
    r = tiny_result.resampling
    after = r["after"]["smotenc"]
    assert after["left"] == after["stayed"] == r["fold_train_original"]["stayed"]
    assert r["fold_validation"]["left"] < r["fold_train_original"]["left"]


def test_bundle_roundtrip(tiny_result, tmp_path, split):
    X_train, X_test, y_train, y_test = split
    run_dir = save_bundle(tiny_result, tmp_path / "run")
    b = load_bundle(run_dir)
    model = b.load_model()
    p = model.predict_proba(X_test)[:, 1]
    assert np.allclose(p, b.test_predictions["probability"].to_numpy(), atol=1e-6)
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["dataset"]["sha256"] and meta["split"]["seed"] == 42
    assert (run_dir / "config.json").exists() and (run_dir / "cv_folds.csv").exists()


def test_committed_default_model_loads_and_reproduces(split):
    if not (config.DEFAULT_EXPERIMENT_DIR / "selection.json").exists():
        pytest.skip("default experiment not generated")
    _, X_test, _, _ = split
    b = load_bundle()
    model = joblib.load(b.model_path)
    p = model.predict_proba(X_test)[:, 1]
    saved = b.test_predictions.set_index("row_id").loc[X_test.index, "probability"].to_numpy()
    assert np.allclose(p, saved, atol=1e-5)
    m = compute_metrics(b.test_predictions["actual_left"], b.test_predictions["probability"], b.threshold)
    assert m["pr_auc"] == pytest.approx(b.test_metrics["pr_auc"], abs=1e-6)


def test_risk_bands():
    assert risk_band(0.50, 0.36) == "high"
    assert risk_band(0.20, 0.36) == "elevated"
    assert risk_band(0.10, 0.36) == "low"


def test_inference_validation(split):
    X_train, X_test, y_train, _ = split
    rec = typical_employee(X_train)
    assert validate_employee(rec) == []
    bad = dict(rec, YearsAtCompany=rec["TotalWorkingYears"] + 5)
    assert any("cannot exceed" in e for e in validate_employee(bad))
    assert validate_employee(dict(rec, OverTime="Sometimes"))
    assert validate_employee(dict(rec, Age=12))
    missing = {k: v for k, v in rec.items() if k != "JobRole"}
    assert validate_employee(missing)
    pipe = build_pipeline("logreg", "none", 42, y_train).fit(X_train, y_train)
    res = predict_employee(pipe, rec, 0.36)
    assert 0 <= res.probability <= 1 and res.band in {"low", "elevated", "high"}
    with pytest.raises(ValueError):
        predict_employee(pipe, bad, 0.36)


def test_every_real_record_passes_input_validation(df):
    X = df[schema.MODEL_FEATURES]
    for _, r in X.iterrows():
        assert validate_employee({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}) == []
