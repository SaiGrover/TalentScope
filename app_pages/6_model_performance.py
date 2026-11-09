"""Model performance: cross-validated comparison, then the single held-out evaluation."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from talentscope.evaluation import METRIC_LABELS, calibration_table, compute_metrics, curve_points, threshold_sweep
from talentscope.explainability import load_explanations
from talentscope.modeling import MODEL_SPECS, STRATEGIES
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import bundle, run_picker

ui.page_header(
    "06 · Model", "Model performance",
    "Two kinds of evidence, kept apart. <b>Cross-validated</b> results come from the training split and were used to "
    "choose the model and threshold. The <b>held-out</b> result is the selected model scored once on 294 employees "
    "it never saw.")
c_pick, c_info = st.columns([1, 2.4], vertical_alignment="bottom")
with c_pick:
    run = run_picker("perf_run")
b = bundle(run) if run else None
if b is None:
    ui.empty_state("No experiment found", "Run `python scripts/train_default.py`, or launch one in the Model lab.",
                   ":material/model_training:")
    st.stop()
with c_info:
    st.caption(f"Run created {b.metadata['created_utc']} · dataset sha256 `{b.metadata['dataset']['sha256'][:12]}…` · "
               f"seed {b.config['seed']} · {b.config['cv_folds']}-fold CV x {b.config['cv_repeats']} repeats · "
               f"{b.config['n_iter']} search candidates per configuration")

sel = b.selection
cv = b.cv_summary.copy()
tm = b.test_metrics
boot = b.test_bootstrap.set_index("metric")
prev = b.metadata["split"]["test_positives"] / b.metadata["split"]["test_rows"]
ui.tiles([
    ("Selected", sel["model_name"], f"{sel['strategy_name']} · {sel['rule'].replace('_', '-')} rule", T.BLUE),
    ("CV PR-AUC", f"{sel['cv_pr_auc_mean']:.3f}", f"± {sel['cv_pr_auc_std']:.3f} SD over folds · training data", T.TEAL),
    ("Held-out PR-AUC", f"{tm['pr_auc']:.3f}", f"95% CI {boot.loc['pr_auc', 'ci_low']:.3f} to {boot.loc['pr_auc', 'ci_high']:.3f} · chance {prev:.3f}", T.CLAY),
    ("Held-out ROC-AUC", f"{tm['roc_auc']:.3f}", f"95% CI {boot.loc['roc_auc', 'ci_low']:.3f} to {boot.loc['roc_auc', 'ci_high']:.3f}", T.CLAY),
    ("Threshold", f"{b.threshold:.2f}", f"max {sel['threshold_metric'].upper()} on out-of-fold predictions", T.AMBER),
    ("Brier score", f"{tm['brier']:.3f}", f"held-out · prevalence-only {prev * (1 - prev):.3f}", T.VIOLET),
])
ui.note(f"<b>Why this model.</b> {sel['reason']}", T.BLUE)

# ----------------------------------------------------------------------------- CV comparison
ui.section("Cross-validated comparison", "training split only · these numbers chose the model")
metric = st.segmented_control("Metric", ["pr_auc", "roc_auc", "f1", "recall", "precision", "balanced_accuracy", "brier"],
                              default="pr_auc", format_func=lambda m: METRIC_LABELS[m], key="perf_metric")
metric = metric or "pr_auc"
lower_better = metric == "brier"
cv = cv.sort_values(f"cv_{metric}_mean", ascending=lower_better).reset_index(drop=True)
labels = [("★ " if s else "") + f"{T.MODEL_SHORT.get(m, m)} · {n}"
          for m, n, s in zip(cv["model_name"], cv["strategy_name"], cv["selected"])]
fig = go.Figure()
for skey, strat in STRATEGIES.items():
    m = cv["strategy"] == skey
    if not m.any():
        continue
    fig.add_trace(go.Scatter(
        x=cv.loc[m, f"cv_{metric}_mean"], y=[labels[i] for i in np.flatnonzero(m)], mode="markers", name=strat.name,
        error_x=dict(type="data", array=cv.loc[m, f"cv_{metric}_std"], color="rgba(170,196,206,.45)", thickness=1.5, width=0),
        marker=dict(size=[14 if s else 10 for s in cv.loc[m, "selected"]], color=T.STRATEGY_COLORS[skey],
                    symbol=T.STRATEGY_SYMBOLS[skey], line=dict(color=T.PANEL, width=1.5)),
        hovertemplate="%{y}<br>" + METRIC_LABELS[metric] + " %{x:.3f} ± %{error_x.array:.3f}<extra></extra>"))
fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], showgrid=False)
fig.update_xaxes(title=f"{METRIC_LABELS[metric]}, mean ± SD across {int(cv['n_folds'].iloc[0])} folds"
                 + (" (threshold 0.5 per fold)" if metric in ("f1", "recall", "precision", "balanced_accuracy") else ""))
ui.chart(C.style(fig, max(320, 24 * len(cv) + 110)), "Every configuration, ranked",
         "★ = selected · colour and marker = imbalance strategy", key="perf_rank",
         what="Each row is one algorithm trained with one imbalance strategy, tuned by randomised search and then "
              "re-scored on repeated stratified folds of the training set. Whiskers show the fold-to-fold SD.",
         why="Overlapping whiskers mean the data cannot separate those configurations; that is why the selection rule "
             "prefers the simplest one within one standard error of the best. Threshold-based metrics here use 0.5.")

c1, c2 = st.columns([1, 1.1], gap="medium")
with c1:
    piv = cv.pivot_table(index="model_name", columns="strategy_name", values="cv_pr_auc_mean")
    piv = piv.reindex(columns=[s.name for s in STRATEGIES.values() if s.name in piv.columns])
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    ui.chart(C.heatmap(piv, colorscale=T.SEQ_TEAL, text_fmt=".3f", colorbar_title="PR-AUC", hover_label="CV PR-AUC",
                       height=60 * len(piv) + 120),
             "Imbalance strategy x algorithm", "mean CV PR-AUC", key="perf_heat",
             what="Mean cross-validated PR-AUC for every algorithm under every imbalance strategy.",
             why="Read across a row to see what rebalancing did for that algorithm.")
with c2:
    rows = []
    for m in cv["model"].unique():
        base = cv[(cv["model"] == m) & (cv["strategy"] == "none")]
        if base.empty:
            continue
        for _, r in cv[(cv["model"] == m) & (cv["strategy"] != "none")].iterrows():
            se = np.sqrt((r["cv_pr_auc_std"] ** 2 + base["cv_pr_auc_std"].iloc[0] ** 2) / r["n_folds"])
            d = r["cv_pr_auc_mean"] - base["cv_pr_auc_mean"].iloc[0]
            rows.append({"Algorithm": r["model_name"], "Strategy": r["strategy_name"], "Δ PR-AUC vs none": d,
                         "Δ recall @0.5": r["cv_recall_mean"] - base["cv_recall_mean"].iloc[0],
                         "Verdict": "better" if d > 2 * se else "worse" if d < -2 * se else "no clear difference"})
    with ui.card("Did rebalancing help?", "change against the same algorithm with no rebalancing", key="perf_delta"):
        if rows:
            dd = pd.DataFrame(rows)
            st.dataframe(dd, hide_index=True, alt="Rebalancing deltas",
                         column_config={"Δ PR-AUC vs none": st.column_config.NumberColumn(format="%+.3f"),
                                        "Δ recall @0.5": st.column_config.NumberColumn(format="%+.3f")})
            smote = dd[dd["Strategy"] == STRATEGIES["smotenc"].name]
            if len(smote):
                st.caption(f"SMOTENC: {int((smote['Verdict'] == 'better').sum())} better, "
                           f"{int((smote['Verdict'] == 'worse').sum())} worse, "
                           f"{int((smote['Verdict'] == 'no clear difference').sum())} no clear difference in PR-AUC "
                           f"(out of {len(smote)} algorithms). 'Clear' means beyond ±2 approximate standard errors. "
                           "Rebalancing mostly shifts predicted probabilities upwards, which raises recall at a fixed 0.5 "
                           "threshold; PR-AUC is threshold-free, so it shows whether ranking quality actually changed.")
        else:
            st.caption("Run 'No rebalancing' alongside another strategy to compare.")

top = cv.head(8)["combo"].tolist()
folds = b.cv_folds[b.cv_folds["combo"].isin(top)]
fig = go.Figure()
for cid in top:
    r = cv[cv["combo"] == cid].iloc[0]
    f = folds[folds["combo"] == cid]
    fig.add_trace(go.Box(y=f[metric], name=f"{T.MODEL_SHORT.get(r['model_name'], r['model_name'])}<br>{r['strategy_name']}", boxpoints="all", jitter=0.4,
                         pointpos=0, marker=dict(size=6, color=T.STRATEGY_COLORS[r["strategy"]], line=dict(color=T.PANEL, width=1)),
                         line=dict(color=T.MUT, width=1.2), fillcolor="rgba(0,0,0,0)", showlegend=False,
                         hovertemplate="%{y:.3f}<extra></extra>"))
fig.update_yaxes(title=f"fold {METRIC_LABELS[metric]}")
ui.chart(C.style(fig, 360), "Fold-to-fold variability", "top eight configurations, every fold shown", key="perf_box",
         what="Each dot is the score on one validation fold (about 235 employees, of whom about 38 left).",
         why="With so few leavers per fold, scores move by several points between folds. Differences smaller than "
             "this spread are not evidence that one configuration is better.")

ui.section("Out-of-fold curves", "training split · probabilities averaged over CV repeats")
choices = cv["combo"].tolist()
default = [sel["combo"]] + [c for c in choices if c != sel["combo"] and c.endswith("|none")][:2]
picked = st.multiselect("Configurations", choices, default=default, max_selections=4, key="perf_curves",
                        format_func=lambda c: f"{MODEL_SPECS[c.split('|')[0]].name} · {STRATEGIES[c.split('|')[1]].name}")
y_tr = b.oof["actual_left"].to_numpy()
palette = [T.TEAL, T.BLUE, T.AMBER, T.VIOLET]
roc, pr = [], []
for i, cid in enumerate(picked):
    cp = curve_points(y_tr, b.oof[cid].to_numpy())
    name = f"{MODEL_SPECS[cid.split('|')[0]].name} · {STRATEGIES[cid.split('|')[1]].name}"
    roc.append({"name": name, "x": cp["fpr"], "y": cp["tpr"], "color": palette[i]})
    pr.append({"name": name, "x": cp["recall"], "y": cp["precision"], "color": palette[i]})
c1, c2 = st.columns(2, gap="medium")
with c1:
    ui.chart(C.curve_chart(roc, "roc"), "ROC curves (cross-validated)", key="perf_roc_cv")
with c2:
    ui.chart(C.curve_chart(pr, "pr", baseline=float(y_tr.mean())), "Precision-recall curves (cross-validated)",
             key="perf_pr_cv",
             what="Precision against recall as the threshold moves. The flat line is the precision of random guessing.",
             why="With 16% leavers, PR curves show the cost of finding more leavers more honestly than ROC curves.")

# ----------------------------------------------------------------------------- held-out
ui.section("Held-out evaluation", f"selected model only · {b.metadata['split']['test_rows']} employees · scored once")
tp = b.test_predictions
c1, c2, c3 = st.columns([1, 1, 0.9], gap="medium")
with c1:
    cp = b.test_curves
    oofc = curve_points(y_tr, b.oof[sel["combo"]].to_numpy())
    ui.chart(C.curve_chart([{"name": "held-out test", "x": cp["fpr"], "y": cp["tpr"], "color": T.CLAY},
                            {"name": "cross-validated", "x": oofc["fpr"], "y": oofc["tpr"], "color": T.TEAL,
                             "dash": "dot", "width": 1.5}], "roc"),
             "ROC: held-out vs CV", key="perf_roc_test")
with c2:
    ui.chart(C.curve_chart([{"name": "held-out test", "x": cp["recall"], "y": cp["precision"], "color": T.CLAY},
                            {"name": "cross-validated", "x": oofc["recall"], "y": oofc["precision"], "color": T.TEAL,
                             "dash": "dot", "width": 1.5}], "pr", baseline=prev),
             "Precision-recall: held-out vs CV", key="perf_pr_test",
             what="The selected model's curve on the test set (solid) against its cross-validated curve (dotted).",
             why="Close curves suggest the CV estimate was not over-optimistic. A small test set makes the solid "
                 "curve jagged; see the bootstrap intervals.")
with c3:
    ui.chart(C.confusion_chart(tm["tn"], tm["fp"], tm["fn"], tm["tp"]), f"Confusion matrix at {b.threshold:.2f}",
             key="perf_cm",
             what=f"Test employees by actual and predicted outcome at the threshold chosen on training data.",
             why=f"Of {tm['tp'] + tm['fn']} actual leavers, {tm['tp']} were flagged. {tm['fp']} stayers were flagged too.")

c1, c2 = st.columns([1.1, 1], gap="medium")
with c1:
    bt = b.test_bootstrap.copy()
    bt = bt[bt["metric"] != "brier"]
    fig = go.Figure(go.Scatter(
        x=bt["estimate"], y=bt["label"], mode="markers",
        error_x=dict(type="data", symmetric=False, array=bt["ci_high"] - bt["estimate"],
                     arrayminus=bt["estimate"] - bt["ci_low"], color="rgba(217,99,74,.6)", thickness=2, width=0),
        marker=dict(size=11, color=T.CLAY, line=dict(color=T.PANEL, width=2)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="held-out estimate with 95% bootstrap interval")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    ui.chart(C.style(fig, 330), "Held-out metrics with uncertainty", key="perf_forest",
             what=f"Point estimates on the test set with 95% percentile intervals from {b.config['bootstrap_resamples']:,} "
                  "stratified bootstrap resamples.",
             why="The test set holds only about 47 leavers, so every held-out metric is uncertain by several points. "
                 "Compare models on the CV results, and quote these with their intervals.")
with c2:
    with ui.card("Classification report", f"held-out · threshold {b.threshold:.2f}", key="perf_report"):
        st.dataframe(b.classification_report.round(3), hide_index=True, alt="Classification report")
        m05 = b.metadata.get("test_metrics_at_0_5", {})
        if m05:
            st.caption(f"At the default 0.5 threshold the same model would have recall {m05['recall']:.2f} and "
                       f"precision {m05['precision']:.2f}; the tuned threshold trades between them deliberately.")

c1, c2 = st.columns(2, gap="medium")
with c1:
    series = [{"name": "cross-validated (10 bins)", "x": b.oof_calibration["mean_predicted"],
               "y": b.oof_calibration["observed_rate"], "color": T.TEAL},
              {"name": "held-out (5 bins)", "x": b.test_calibration["mean_predicted"],
               "y": b.test_calibration["observed_rate"], "color": T.CLAY}]
    ui.chart(C.calibration_chart(series), "Calibration", "quantile bins", key="perf_cal",
             what="Mean predicted probability against the observed attrition rate in bins of employees.",
             why="Points below the diagonal mean the model over-states risk, which rebalancing tends to do. The "
                 f"held-out Brier score is {tm['brier']:.3f}; a model that always predicts the base rate scores "
                 f"{prev * (1 - prev):.3f}.")
with c2:
    sweep = threshold_sweep(y_tr, b.oof[sel["combo"]].to_numpy())
    explore = st.slider("Explore a threshold (out-of-fold, training)", 0.05, 0.95, float(b.threshold), 0.01,
                        key="perf_thr")
    mm = compute_metrics(y_tr, b.oof[sel["combo"]].to_numpy(), explore)
    ui.chart(C.threshold_chart(sweep, b.threshold, marker=explore), "Threshold trade-off", key="perf_thr_chart",
             what="Precision, recall and F1 on out-of-fold training predictions as the decision threshold moves. "
                  "The white rule is the selected threshold; the amber rule is the one you are exploring.",
             why=f"At {explore:.2f}: precision {mm['precision']:.2f}, recall {mm['recall']:.2f}, "
                 f"{(mm['tp'] + mm['fp']) / len(y_tr):.0%} of employees flagged. The test set was never used to "
                 "choose the threshold.")

# ----------------------------------------------------------------------------- resampling evidence
ui.section("Resampling evidence", "class counts inside one training fold, before and after the sampler")
r = b.resampling
rows = [("Training split", r["train_set"]), ("CV training fold (before)", r["fold_train_original"])]
for k_, v in r.get("after", {}).items():
    rows.append((f"CV training fold after {STRATEGIES[k_].name}", v))
rows.append(("CV validation fold (never resampled)", r["fold_validation"]))
rows.append(("Held-out test set (never resampled)", {"stayed": int((tp["actual_left"] == 0).sum()),
                                                     "left": int((tp["actual_left"] == 1).sum())}))
fig = go.Figure()
names = [n for n, _ in rows]
for key, name, color in (("stayed", "Stayed", T.STAYED), ("left", "Left", T.LEFT)):
    vals = [v[key] for _, v in rows]
    fig.add_trace(go.Bar(y=names, x=vals, orientation="h", name=name, marker=dict(color=color, line=dict(color=T.PANEL, width=2),
                                                                                   cornerradius=4),
                         text=[f"{v:,}" for v in vals], textposition="inside", insidetextanchor="middle",
                         textfont=dict(color="#0b1115", size=11)))
fig.update_layout(barmode="stack")
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(title="employees (synthetic rows included after resampling)")
ui.chart(C.style(fig, 60 * len(rows) + 90), "Where resampling happens", key="perf_resample",
         what="Class counts at each stage. SMOTENC and random oversampling only change the training part of a fold.",
         why="Resampling before splitting would put synthetic copies of validation employees into training and "
             "inflate every score. The pipeline makes that impossible.")

# ----------------------------------------------------------------------------- responsible AI
ui.section("Responsible AI review", "held-out error rates by group · selected threshold")
ex = load_explanations(b)
gm = ex.get("group_metrics")
if gm is None or gm.empty:
    st.caption("Group metrics are computed when a run's explanations are generated.")
else:
    attr = st.segmented_control("Attribute", gm["attribute"].unique().tolist(), default="Gender", key="perf_attr")
    g = gm[gm["attribute"] == (attr or "Gender")].copy()
    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        fig = go.Figure()
        for col, name, color in (("observed_rate", "Observed attrition", T.NEUTRAL), ("flagged_rate", "Flagged by model", T.CLAY),
                                 ("recall_tpr", "Recall (leavers caught)", T.TEAL),
                                 ("false_positive_rate", "False-positive rate", T.AMBER)):
            fig.add_trace(go.Bar(x=g["group"], y=g[col], name=name, marker=dict(color=color, cornerradius=4),
                                 hovertemplate="%{x}<br>" + name + " %{y:.1%}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25)
        fig.update_yaxes(tickformat=".0%", title="share of group")
        ui.chart(C.style(fig, 340), f"Error rates by {attr or 'Gender'}", key="perf_fair",
                 what="For each group in the test set: the observed attrition rate, the share the model flags, the share "
                      "of actual leavers it catches and the share of stayers it wrongly flags.",
                 why="Large gaps in recall or false-positive rate would mean the model's mistakes fall unevenly. Groups "
                     "are small, so treat gaps of a few points as noise.")
    with c2:
        with ui.card("Group table", key="perf_fair_tbl"):
            st.dataframe(g.drop(columns=["attribute"]).round(3), hide_index=True, alt="Group metrics")
ui.note("<b>Limitations.</b> IBM's data is fictional and small (237 leavers in total, about 47 in the test set). "
        "The model learns associations, not causes. It includes gender, age and marital status because the dataset "
        "does; any real use would need a review of whether those attributes should be used at all, a fairness "
        "assessment on real outcomes, and human judgement on every individual case. Scores are not an employment "
        "decision.", T.AMBER)

with st.expander("Search trials and tuned parameters", icon=":material/tune:"):
    st.json(b.best_params, expanded=False)
    if not b.search_trials.empty:
        st.dataframe(b.search_trials.sort_values("mean_cv_pr_auc", ascending=False), hide_index=True, height=360,
                     alt="Search trials")
    st.download_button("Download CV summary (CSV)", b.cv_summary.to_csv(index=False).encode("utf-8"),
                       "talentscope_cv_summary.csv", "text/csv", icon=":material/download:", key="perf_dl")
