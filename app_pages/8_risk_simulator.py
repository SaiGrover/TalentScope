"""Attrition risk simulator: score a hypothetical employee and see why."""
import numpy as np
import pandas as pd
import streamlit as st

from talentscope import config
from talentscope.analytics import rate_by
from talentscope.data import schema
from talentscope.evaluation import wilson_interval
from talentscope.explainability import aggregate_by_source, build_explainer, explain
from talentscope.modeling.inference import predict_employee, risk_band, to_frame, typical_employee, validate_employee
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import bundle, dataset, model, run_picker, split_frames

ui.page_header(
    "08 · Act", "Attrition risk simulator",
    "Describe an employee and the selected model returns an estimated probability of leaving, the attributes that "
    "drove it, and how reliable estimates in that range have been. This is a demonstration on synthetic data. It "
    "is not an employment decision and must not be used to treat anyone differently.")
c_pick, _ = st.columns([1, 2.4])
with c_pick:
    run = run_picker("sim_run")
b = bundle(run) if run else None
if b is None or b.selection["model"] == "dummy":
    ui.empty_state("No usable model", "Train the default model or launch a run in the Model lab.", ":material/model_training:")
    st.stop()
try:
    mdl = model(b)
except Exception as exc:  # incompatible pickle, missing file
    st.error(f"The saved model could not be loaded ({type(exc).__name__}). Re-run `python scripts/train_default.py`.",
             icon=":material/error:")
    st.stop()

X_train, X_test, y_train, y_test = split_frames(b.config["seed"], b.config["test_size"])
df = dataset()


@st.cache_resource(show_spinner=False, max_entries=4)
def get_explainer(run_dir: str, mtime: float, model_key: str):
    return build_explainer(model(b), model_key, X_train)


# ----------------------------------------------------------------------------- presets
if "sim_rec" not in st.session_state:
    st.session_state["sim_rec"] = typical_employee(X_train)
tp = b.test_predictions.set_index("row_id")


def load_preset(rec: dict) -> None:
    st.session_state["sim_rec"] = rec
    for f, v in rec.items():
        st.session_state[f"sim_{f}"] = v


with st.container(horizontal=True, vertical_alignment="bottom"):
    if st.button("Typical employee", icon=":material/person:", key="sim_typ",
                 help="Median numeric values and the most common category from the training split."):
        load_preset(typical_employee(X_train))
    pick = st.selectbox("Or start from a held-out employee", tp.sort_values("probability", ascending=False).index,
                        format_func=lambda r: f"#{int(tp.loc[r, 'EmployeeNumber'])} · model risk {tp.loc[r, 'probability']:.0%}",
                        key="sim_pick", index=None, placeholder="Choose a test-set employee", width=320)
    if st.button("Load", icon=":material/upload:", key="sim_load", disabled=pick is None):
        load_preset({f: (X_test.loc[pick, f].item() if hasattr(X_test.loc[pick, f], "item") else X_test.loc[pick, f])
                     for f in schema.MODEL_FEATURES})
rec0 = st.session_state["sim_rec"]
for f, v in rec0.items():
    st.session_state.setdefault(f"sim_{f}", v)

roles_by_dept = df.groupby("Department")["JobRole"].unique().apply(sorted).to_dict()

c_form, c_out = st.columns([1.15, 1], gap="large")
with c_form:
    with st.form("sim_form", border=False):
        with ui.card("Role and organisation", key="sim_c1"):
            a, b_, c = st.columns(3)
            a.selectbox("Department", schema.EXPECTED_CATEGORIES["Department"], key="sim_Department")
            b_.selectbox("Job role", schema.EXPECTED_CATEGORIES["JobRole"], key="sim_JobRole")
            c.selectbox("Job level", [1, 2, 3, 4, 5], key="sim_JobLevel", format_func=lambda v: schema.ORDINAL_LEVELS["JobLevel"][v])
            a, b_, c = st.columns(3)
            a.selectbox("Overtime", ["No", "Yes"], key="sim_OverTime")
            b_.selectbox("Business travel", schema.EXPECTED_CATEGORIES["BusinessTravel"], key="sim_BusinessTravel",
                         format_func=lambda s: s.replace("_", " "))
            c.number_input("Distance from home", 1, 30, key="sim_DistanceFromHome")
        with ui.card("Compensation", key="sim_c2"):
            a, b_, c = st.columns(3)
            a.number_input("Monthly income", 1000, 20000, step=100, key="sim_MonthlyIncome")
            b_.number_input("Last salary hike %", 0, 30, key="sim_PercentSalaryHike")
            c.selectbox("Stock option level", [0, 1, 2, 3], key="sim_StockOptionLevel")
            a, b_, c = st.columns(3)
            a.number_input("Daily rate", 100, 1500, key="sim_DailyRate")
            b_.number_input("Hourly rate", 30, 100, key="sim_HourlyRate")
            c.number_input("Monthly rate", 2000, 27000, step=100, key="sim_MonthlyRate")
        with ui.card("Experience and tenure", key="sim_c3"):
            a, b_, c = st.columns(3)
            a.number_input("Age", 18, 65, key="sim_Age")
            b_.number_input("Total working years", 0, 45, key="sim_TotalWorkingYears")
            c.number_input("Companies worked for", 0, 10, key="sim_NumCompaniesWorked")
            a, b_, c = st.columns(3)
            a.number_input("Years at company", 0, 45, key="sim_YearsAtCompany")
            b_.number_input("Years in current role", 0, 45, key="sim_YearsInCurrentRole")
            c.number_input("Years with manager", 0, 45, key="sim_YearsWithCurrManager")
            a, b_, c = st.columns(3)
            a.number_input("Years since promotion", 0, 45, key="sim_YearsSinceLastPromotion")
            b_.number_input("Trainings last year", 0, 6, key="sim_TrainingTimesLastYear")
            c.selectbox("Performance rating", [1, 2, 3, 4], key="sim_PerformanceRating",
                        format_func=lambda v: schema.ORDINAL_LEVELS["PerformanceRating"][v])
        with ui.card("Engagement survey", key="sim_c4"):
            lv = lambda col: (lambda v: schema.ORDINAL_LEVELS[col][v])  # noqa: E731
            a, b_, c = st.columns(3)
            a.selectbox("Job satisfaction", [1, 2, 3, 4], key="sim_JobSatisfaction", format_func=lv("JobSatisfaction"))
            b_.selectbox("Environment satisfaction", [1, 2, 3, 4], key="sim_EnvironmentSatisfaction",
                         format_func=lv("EnvironmentSatisfaction"))
            c.selectbox("Relationship satisfaction", [1, 2, 3, 4], key="sim_RelationshipSatisfaction",
                        format_func=lv("RelationshipSatisfaction"))
            a, b_, c = st.columns(3)
            a.selectbox("Job involvement", [1, 2, 3, 4], key="sim_JobInvolvement", format_func=lv("JobInvolvement"))
            b_.selectbox("Work-life balance", [1, 2, 3, 4], key="sim_WorkLifeBalance", format_func=lv("WorkLifeBalance"))
        with ui.card("Personal", "present in the dataset; see the note on sensitive attributes", key="sim_c5"):
            a, b_, c = st.columns(3)
            a.selectbox("Gender", ["Female", "Male"], key="sim_Gender")
            b_.selectbox("Marital status", schema.EXPECTED_CATEGORIES["MaritalStatus"], key="sim_MaritalStatus")
            c.selectbox("Education", [1, 2, 3, 4, 5], key="sim_Education", format_func=lv("Education"))
            st.selectbox("Education field", schema.EXPECTED_CATEGORIES["EducationField"], key="sim_EducationField")
        go_ = st.form_submit_button("Estimate attrition risk", type="primary", icon=":material/insights:", width="stretch")

rec = {f: st.session_state.get(f"sim_{f}") for f in schema.MODEL_FEATURES}
with c_out:
    errors = validate_employee(rec)
    if rec["JobRole"] not in roles_by_dept.get(rec["Department"], []):
        errors.append(f"In this dataset '{rec['JobRole']}' does not occur in {rec['Department']} "
                      f"(it occurs in {', '.join(d for d, r in roles_by_dept.items() if rec['JobRole'] in r)}).")
    if errors:
        with ui.card("Check the inputs", key="sim_err"):
            for e in errors:
                st.markdown(f":material/error: {e}")
            st.caption("The estimate is shown only for a consistent record.")
        st.stop()

    res = predict_employee(mdl, rec, b.threshold)
    colour = {"high": T.CLAY, "elevated": T.AMBER, "low": T.TEAL}[res.band]
    with ui.card("Estimated probability of leaving", f"{b.label} · threshold {b.threshold:.2f} chosen on training data",
                 key="sim_res"):
        ui.tiles([("Probability", f"{res.probability:.0%}", "model estimate", colour),
                  ("Risk band", res.band.capitalize(), f"low < {res.low_cut:.0%} ≤ elevated < {res.threshold:.0%} ≤ high",
                   colour)])
        ui.meter(res.probability, res.threshold, res.low_cut, colour)
        pct = float((b.test_predictions["probability"] < res.probability).mean())
        oof = b.oof[b.selection["combo"]].to_numpy()
        ya = b.oof["actual_left"].to_numpy()
        lo_b, hi_b = max(0, res.probability - 0.05), min(1, res.probability + 0.05)
        m = (oof >= lo_b) & (oof <= hi_b)
        n_bin, k_bin = int(m.sum()), int(ya[m].sum())
        st.markdown(f"- Higher than **{pct:.0%}** of held-out employees' scores.")
        if n_bin >= 10:
            wl, wh = wilson_interval(k_bin, n_bin)
            st.markdown(f"- **Uncertainty.** Of the {n_bin} training employees whose cross-validated score fell between "
                        f"{lo_b:.0%} and {hi_b:.0%}, **{k_bin / n_bin:.0%}** actually left (95% CI {wl:.0%} to {wh:.0%}).")
        else:
            st.markdown(f"- **Uncertainty.** Only {n_bin} training employees received scores near {res.probability:.0%}; "
                        "estimates this far from the bulk of the data are unreliable.")
        r = rate_by(df[(df["JobRole"] == rec["JobRole"]) & (df["OverTime"] == rec["OverTime"])], "JobRole")
        if not r.empty:
            g = r.iloc[0]
            st.markdown(f"- **Context.** In the dataset, {g['rate']:.0%} of {rec['JobRole']}s "
                        f"{'working' if rec['OverTime'] == 'Yes' else 'not working'} overtime left "
                        f"(n = {int(g['n'])}, 95% CI {g['ci_low']:.0%} to {g['ci_high']:.0%}).")

    with st.spinner("Explaining the estimate..."):
        exp_ = get_explainer(str(b.run_dir), b.model_path.stat().st_mtime, b.selection["model"])
        e = explain(mdl, exp_, to_frame(rec), b.selection["model"], kernel_nsamples=300)
    contrib = aggregate_by_source(e.values, e.feature_names).iloc[0].rename(index=schema.PRETTY_NAMES)
    ui.chart(C.waterfall(contrib, e.base_value, e.base_value + contrib.sum(), e.output_space, k=8),
             "What drove this estimate", f"SHAP contributions in {e.output_space}; clay raises risk, teal lowers it",
             key="sim_wf",
             what="The model's average output plus each attribute's contribution for this record.",
             why="Change one input above and re-estimate to see how the model responds. That shows the model's "
                 "sensitivity, not what would happen to a real person.")

ui.note("<b>Use and limits.</b> The model was trained on IBM's fictional sample and reaches a held-out PR-AUC of "
        f"{b.test_metrics['pr_auc']:.2f} (95% CI {b.test_bootstrap.set_index('metric').loc['pr_auc', 'ci_low']:.2f} to "
        f"{b.test_bootstrap.set_index('metric').loc['pr_auc', 'ci_high']:.2f}); at its threshold it catches "
        f"{b.test_metrics['recall']:.0%} of leavers and {b.test_metrics['precision']:.0%} of the people it flags "
        "actually leave. It reflects patterns in that sample, including attributes such as gender, age and marital "
        "status that should not drive decisions about real people. Use the estimate, if at all, to start a "
        "supportive conversation, never to deny opportunities or single anyone out.", T.AMBER)
