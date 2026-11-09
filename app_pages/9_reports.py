"""Reports and insights: downloadable outputs generated from live computations and saved artefacts."""
import streamlit as st

from talentscope.explainability import load_explanations
from talentscope.reporting import (
    attrition_report,
    experiment_comparison_csv,
    model_card,
    model_evaluation_report,
    predictions_csv,
    to_html,
    workforce_report,
)
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import bundle, dataset, filter_bar, filter_summary, run_picker, split_frames

ui.page_header(
    "09 · Act", "Reports and insights",
    "Every report is generated on demand from the loaded data and the saved experiment, in Markdown or as a "
    "self-contained HTML page you can print to PDF. Analytics reports follow the shared filters.")

df_all = dataset()
df = filter_bar(df_all)
note = filter_summary(df_all, df)
st.caption(note)
c_pick, _ = st.columns([1, 2.4])
with c_pick:
    run = run_picker("rep_run")
b = bundle(run) if run else None


def offer(name: str, md: str, stem: str, key: str) -> None:
    with st.container(horizontal=True):
        st.download_button("Markdown", md.encode("utf-8"), f"{stem}.md", "text/markdown", icon=":material/download:",
                           key=f"{key}_md")
        st.download_button("HTML", to_html(md, name).encode("utf-8"), f"{stem}.html", "text/html",
                           icon=":material/download:", key=f"{key}_html", type="primary")
    with st.expander("Preview", icon=":material/visibility:"):
        st.markdown(md)


ui.section("Workforce and attrition", "filters apply")
c1, c2 = st.columns(2, gap="medium")
with c1:
    with ui.card("Workforce analytics report", "headcount, demographics, pay, tenure and engagement by group", key="rep_wf"):
        if df.empty:
            st.caption("No employees match the filters.")
        else:
            offer("Workforce analytics report", workforce_report(df, note), "talentscope_workforce_report", "rep_wf")
with c2:
    with ui.card("Attrition analysis report", "rates with Wilson intervals across 12 dimensions, plus significance tests",
                 key="rep_at"):
        if len(df) < 20:
            st.caption("At least 20 employees are needed.")
        else:
            offer("Attrition analysis report", attrition_report(df, note), "talentscope_attrition_report", "rep_at")

ui.section("Model", "from the selected experiment's saved artefacts")
if b is None:
    ui.empty_state("No experiment found", "Train the default model or launch a run in the Model lab.", ":material/model_training:")
    st.stop()
ex = load_explanations(b)
c1, c2 = st.columns(2, gap="medium")
with c1:
    with ui.card("Model evaluation summary", "protocol, held-out metrics with intervals, CV comparison, SMOTENC effect",
                 key="rep_ev"):
        offer("Model evaluation summary", model_evaluation_report(b, ex), "talentscope_model_evaluation", "rep_ev")
with c2:
    with ui.card("Model card", "intended use, data, evaluation, ethical considerations and caveats", key="rep_mc"):
        offer("Model card", model_card(b, ex), "talentscope_model_card", "rep_mc")

c1, c2 = st.columns(2, gap="medium")
with c1:
    with ui.card("Experiment comparison", "every algorithm x strategy with CV mean and SD", key="rep_cmp"):
        st.download_button("Download CSV", experiment_comparison_csv(b), "talentscope_experiment_comparison.csv",
                           "text/csv", icon=":material/download:", key="rep_cmp_dl", type="primary")
        st.caption(f"{len(b.cv_summary)} configurations from `{b.run_dir.name}`.")
with c2:
    with ui.card("Employee-level predictions", "held-out synthetic employees only", key="rep_pred"):
        X_train, X_test, _, _ = split_frames(b.config["seed"], b.config["test_size"])
        st.download_button("Download CSV", predictions_csv(b, X_test), "talentscope_test_predictions.csv", "text/csv",
                           icon=":material/download:", key="rep_pred_dl", type="primary")
        st.caption(f"{len(b.test_predictions)} rows: probability, risk band and actual outcome for the test set. "
                   "Training-set predictions are not offered because in-sample scores overstate accuracy.")

ui.note("<b>Provenance.</b> Reports carry their generation time, the TalentScope version and, for model reports, the "
        "dataset hash, seeds and library versions of the run they describe.", T.BLUE)
