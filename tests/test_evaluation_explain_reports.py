"""Metrics, bootstrap intervals, SHAP additivity, analytics and report generation."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from talentscope.analytics import association_ranking, headline_findings, rate_by, workforce_kpis
from talentscope.evaluation import bootstrap_ci, compute_metrics, select_threshold, wilson_interval
from talentscope.explainability import aggregate_by_source, build_explainer, explain
from talentscope.modeling import build_pipeline


def test_compute_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.5
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))


def test_select_threshold_maximises_f1():
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.linspace(0, 0.3, 80), np.linspace(0.25, 0.9, 20)])
    t, f1 = select_threshold(y, p)
    assert 0.25 <= t <= 0.31
    assert f1 > 0.9


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    ci = bootstrap_ci(y, p, 0.5, n_resamples=200).set_index("metric")
    for m in ("pr_auc", "roc_auc", "f1"):
        assert ci.loc[m, "ci_low"] <= ci.loc[m, "estimate"] <= ci.loc[m, "ci_high"]


def test_wilson_interval():
    lo, hi = wilson_interval(237, 1470)
    assert lo < 237 / 1470 < hi
    assert wilson_interval(0, 10)[0] == 0.0


def test_analytics(df):
    k = workforce_kpis(df)
    assert k["headcount"] == 1470 and k["left"] == 237
    r = rate_by(df, "OverTime").set_index("group")
    assert r.loc["Yes", "rate"] > r.loc["No", "rate"]
    assert r["n"].sum() == 1470
    a = association_ranking(df)
    assert len(a) == 30 and a["p_adjusted"].between(0, 1).all()
    assert headline_findings(df, k=3)


@pytest.mark.parametrize("model,strategy", [("logreg", "none"), ("tree", "smotenc"), ("xgboost", "class_weight")])
def test_shap_values_reconstruct_predictions(model, strategy, split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline(model, strategy, 42, y_train).fit(X_train, y_train)
    ex = explain(pipe, build_explainer(pipe, model, X_train.sample(150, random_state=0)), X_test.head(8), model)
    out = ex.base_value + ex.values.sum(axis=1)
    if ex.output_space == "log-odds":
        out = 1 / (1 + np.exp(-out))
    assert np.allclose(out, pipe.predict_proba(X_test.head(8))[:, 1], atol=1e-4)
    agg = aggregate_by_source(ex.values, ex.feature_names)
    assert agg.shape == (8, 30)
    assert np.allclose(agg.sum(axis=1), ex.values.sum(axis=1))


def test_reports_build_from_artefacts(df):
    from talentscope import config
    from talentscope.reporting import attrition_report, model_card, model_evaluation_report, to_html, workforce_report
    w = workforce_report(df)
    a = attrition_report(df)
    assert "Workforce analytics report" in w and "| Department |" in w
    assert "Attrition by overtime" in a and "Benjamini" in a
    html = to_html(a, "Attrition")
    assert html.startswith("<!doctype html>") and "<table>" in html
    if (config.DEFAULT_EXPERIMENT_DIR / "selection.json").exists():
        from talentscope.explainability import load_explanations
        from talentscope.modeling import load_bundle
        b = load_bundle()
        ev = model_evaluation_report(b, load_explanations(b))
        card = model_card(b, load_explanations(b))
        assert f"{b.test_metrics['pr_auc']:.3f}" in ev
        assert "Intended use" in card and "Not intended for decisions" in card
