"""Explainable AI: SHAP and permutation importance for the selected model."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

from talentscope.data import schema
from talentscope.explainability import aggregate_by_source, explain_bundle, load_explanations
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import bundle, dataset, run_picker, split_frames

ui.page_header(
    "07 · Model", "Explainable AI",
    "How the selected model turns 30 attributes into a risk score, measured two independent ways: SHAP values "
    "(each feature's contribution to each prediction) and permutation importance (how much held-out PR-AUC drops "
    "when a feature is shuffled). Both describe the model, not the causes of attrition.")
c_pick, _ = st.columns([1, 2.4])
with c_pick:
    run = run_picker("xai_run")
b = bundle(run) if run else None
if b is None:
    ui.empty_state("No experiment found", "Train the default model or launch a run in the Model lab.", ":material/model_training:")
    st.stop()

ex = load_explanations(b)
if b.selection["model"] == "dummy":
    ui.empty_state("Nothing to explain", "The dummy baseline predicts the base rate for everyone.")
    st.stop()
if "shap" not in ex:
    st.info("This run has no saved explanations yet.", icon=":material/info:")
    if st.button("Compute explanations now", type="primary", icon=":material/psychology:", key="xai_compute"):
        X_train, X_test, y_train, y_test = split_frames(b.config["seed"], b.config["test_size"])
        with st.spinner("Computing SHAP values and permutation importance on the test set..."):
            explain_bundle(b, X_train, X_test, y_test,
                           groups=dataset()[["Gender", "AgeBand", "MaritalStatus", "Department"]], kernel_nsamples=150,
                           n_repeats=10)
        st.rerun()
    st.stop()

sh = ex["shap"]
meta = ex.get("meta", {})
space = sh.output_space
agg = aggregate_by_source(sh.values, sh.feature_names)
shap_rank = agg.abs().mean().sort_values(ascending=False)
perm = ex["permutation_importance"].set_index("feature")
common = [f for f in shap_rank.index if f in perm.index]
rho = spearmanr(shap_rank.reindex(common).values, perm.loc[common, "importance_mean"].values).statistic
ui.tiles([
    ("Model", b.selection["model_name"], b.selection["strategy_name"], T.BLUE),
    ("Explainer", {"linear": "Linear SHAP", "tree": "Tree SHAP", "kernel": "Kernel SHAP"}.get(sh.explainer_kind, sh.explainer_kind),
     f"output in {space}", T.TEAL),
    ("Rows explained", f"{len(sh.values)}", meta.get("explained_set", "held-out test set"), T.TEAL),
    ("Top feature (SHAP)", schema.PRETTY_NAMES.get(shap_rank.index[0], shap_rank.index[0]), "largest mean |SHAP|", T.CLAY),
    ("Top feature (permutation)", perm["label"].iloc[0], f"PR-AUC drop {perm['importance_mean'].iloc[0]:.3f}", T.CLAY),
    ("Rank agreement", f"{rho:.2f}", "Spearman rho, SHAP vs permutation", T.AMBER),
])

ui.section("Global importance", "two independent methods on the same held-out employees")
c1, c2 = st.columns(2, gap="medium")
top_n = 15
with c1:
    s = shap_rank.head(top_n)
    ui.chart(C.ranked_bars([schema.PRETTY_NAMES.get(f, f) for f in s.index], s.values, color=T.TEAL, fmt=".3f",
                           xlabel=f"mean |SHAP| ({space})"),
             "Mean absolute SHAP value", "one-hot columns summed back to their attribute", key="xai_shap",
             what="Average size of each attribute's contribution to a prediction, ignoring direction.",
             why="This is how much the model leans on an attribute, not whether the attribute matters in the world.")
with c2:
    p = perm.head(top_n)
    ui.chart(C.ranked_bars(p["label"], p["importance_mean"], color=T.BLUE, fmt=".3f", errors=p["importance_std"].values,
                           xlabel="drop in held-out PR-AUC when shuffled"),
             "Permutation importance", f"{meta.get('permutation', {}).get('n_repeats', '')} shuffles per attribute, ± SD",
             key="xai_perm",
             what="How much held-out PR-AUC falls when one attribute's values are shuffled across employees.",
             why="Correlated attributes share credit: shuffling income alone barely hurts if job level still carries "
                 "the same information. Small or negative values mean the model does not rely on that attribute.")

ui.section("How feature values push predictions", "SHAP summary, one dot per held-out employee")
imp = sh.importance()
order = [sh.feature_names.index(f) for f in imp["feature"].head(15)]
ui.chart(C.beeswarm(sh.values, sh.display_data, sh.pretty_names, order),
         "SHAP summary", f"top 15 model inputs · x = contribution in {space} · colour = feature value", key="xai_bee",
         what="Each row is a model input (one-hot columns appear separately). Dots to the right raise that employee's "
              "predicted risk, dots to the left lower it. Orange means a high value of the input, blue a low one.",
         why="Reading colour against position gives the direction: for example orange dots on the right of "
             "'Overtime = Yes' mean working overtime raises the model's risk score.")

directions = []
for j in order[:10]:
    v, d = sh.values[:, j], sh.display_data[:, j]
    if np.std(d) > 0:
        r = np.corrcoef(d, v)[0, 1]
        directions.append((sh.pretty_names[j], r))
with ui.card("Directions learned by the model", "correlation between an input's value and its SHAP value", key="xai_dir"):
    lines = [f"- **{n}**: higher values {'raise' if r > 0 else 'lower'} the predicted risk (r = {r:+.2f})"
             for n, r in directions]
    st.markdown("\n".join(lines))
    st.caption("For one-hot inputs, 'higher' means the employee is in that category."
               + (" For a linear model every input's contribution is proportional to its value, so these correlations "
                  "are exactly ±1 by construction; tree models give values in between." if sh.explainer_kind == "linear"
                  else ""))

ui.section("Dependence", "one input's value against its contribution")
c1, c2 = st.columns([1, 2.6], gap="medium")
X_train, X_test, y_train, y_test = split_frames(b.config["seed"], b.config["test_size"])
with c1:
    feat = st.selectbox("Input", imp["feature"].head(25).tolist(), format_func=schema.pretty_feature, key="xai_dep")
    j = sh.feature_names.index(feat)
    st.caption(schema.FEATURE_DESCRIPTIONS.get(schema.source_feature(feat), ""))
with c2:
    actual = y_test.reindex(sh.row_ids).to_numpy()
    fig = go.Figure()
    for val, name, color in ((0, "Stayed", T.STAYED), (1, "Left", T.LEFT)):
        m = actual == val
        jit = np.random.default_rng(1).uniform(-0.12, 0.12, m.sum()) if len(np.unique(sh.display_data[:, j])) <= 6 else 0
        fig.add_trace(go.Scatter(x=sh.display_data[m, j] + jit, y=sh.values[m, j], mode="markers", name=name,
                                 marker=dict(color=color, size=8, opacity=0.75, line=dict(color=T.PANEL, width=1)),
                                 hovertemplate="value %{x:.2f}<br>SHAP %{y:+.3f}<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=T.BORDER_STRONG, line_width=1)
    fig.update_xaxes(title=schema.pretty_feature(feat))
    fig.update_yaxes(title=f"SHAP ({space})")
    ui.chart(C.style(fig, 360), f"Dependence: {schema.pretty_feature(feat)}", "colour = actual outcome", key="xai_depfig",
             what="Each dot is a held-out employee. The vertical position is how much this input moved their prediction.",
             why="For a logistic regression the relationship is a straight line by construction; tree models can bend.")

ui.section("Explain one employee", "held-out test set · synthetic records")
tp = b.test_predictions.set_index("row_id")
row_order = tp.sort_values("probability", ascending=False).index.tolist()
labels = {rid: f"#{int(tp.loc[rid, 'EmployeeNumber'])} · risk {tp.loc[rid, 'probability']:.0%} · "
               f"{'left' if tp.loc[rid, 'actual_left'] else 'stayed'}" for rid in row_order}
c1, c2 = st.columns([1, 2.2], gap="medium")
with c1:
    rid = st.selectbox("Employee (sorted by predicted risk)", row_order, format_func=labels.get, key="xai_emp")
    i = int(np.flatnonzero(sh.row_ids == rid)[0])
    p = tp.loc[rid, "probability"]
    band = "high" if p >= b.threshold else "elevated" if p >= 0.5 * b.threshold else "low"
    ui.tiles([("Predicted risk", f"{p:.0%}", f"{band} · threshold {b.threshold:.2f}",
               T.CLAY if band == "high" else T.AMBER if band == "elevated" else T.TEAL),
              ("Actual outcome", "Left" if tp.loc[rid, "actual_left"] else "Stayed", "known for test employees", T.BLUE)])
    raw = X_test.loc[rid]
    st.dataframe(pd.DataFrame({"value": raw.astype(str)}).rename(index=schema.PRETTY_NAMES), height=360,
                 alt="Employee attributes")
with c2:
    contrib = agg.iloc[i].rename(index=schema.PRETTY_NAMES)
    final = sh.base_value + contrib.sum()
    ui.chart(C.waterfall(contrib, sh.base_value, final, space, k=10),
             "Why this score", f"contributions in {space}; clay raises risk, teal lowers it", key="xai_wf",
             what="Starts at the model's average output and adds each attribute's contribution for this employee, "
                  "largest first. One-hot columns are summed back to their attribute.",
             why=("The total equals the model's output for this employee"
                  + (f" in log-odds; {final:+.2f} log-odds is a probability of {1 / (1 + np.exp(-final)):.0%}."
                     if space == "log-odds" else ".")))
ui.note("<b>Model explanations are not causes.</b> A large SHAP value for overtime means the model's score moves "
        "with overtime in this synthetic sample. It does not show that reducing someone's overtime would keep them, "
        "and correlated attributes (income, job level, age, tenure) split credit between them in ways that depend on "
        "the model.", T.AMBER)
