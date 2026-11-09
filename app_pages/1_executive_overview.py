"""Executive overview: the HR command centre."""
import pandas as pd
import streamlit as st

from talentscope.analytics import headline_findings, rate_by, workforce_kpis
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import bundle, dataset, filter_bar, filter_summary

df_all = dataset()
ui.page_header(
    "01 · Understand", "Executive overview",
    "Headcount, attrition and the groups where leaving concentrates. Every figure is computed from the "
    "loaded dataset on each run, and every rate carries its group size and a 95% Wilson interval.")

df = filter_bar(df_all)
# drill-down: clicking a department bar narrows this page to it
drill = st.session_state.get("ov_drill")
if drill and drill in df["Department"].unique():
    df = df[df["Department"] == drill]
st.caption(filter_summary(df_all, df) + (f" · drilled into **{drill}**" if drill else ""))

if df.empty:
    ui.empty_state("No employees match these filters", "Clear a filter above to see results.", ":material/filter_alt_off:")
    st.stop()

k = workforce_kpis(df)
lo, hi = k["rate_ci"]
overall = workforce_kpis(df_all)["attrition_rate"]
ui.tiles([
    ("Employees", f"{k['headcount']:,}", f"{k['departments']} departments · {k['roles']} roles", T.BLUE),
    ("Attrition rate", f"{k['attrition_rate']:.1%}", f"95% CI {lo:.1%} to {hi:.1%}", T.CLAY),
    ("Left", f"{k['left']:,}", f"{k['stayed']:,} stayed", T.CLAY),
    ("Average tenure", f"{k['avg_tenure']:.1f}", f"years · median {k['median_tenure']:.0f}", T.TEAL, ""),
    ("Monthly income", f"{k['avg_income']:,.0f}", f"mean · median {k['median_income']:,.0f}", T.AMBER),
    ("Overtime", f"{k['overtime_share']:.0%}", f"of employees · mean age {k['avg_age']:.0f}", T.VIOLET),
])

left, right = st.columns([1.25, 1], gap="medium")
with left:
    fig = C.stacked_outcome_bars(df, "Department", title=None)
    fig.update_layout(clickmode="event+select")
    ev = ui.chart(fig, "Headcount by department",
                  "Click a bar to drill into that department; click the selection again or use reset below.",
                  what="Employees per department, split into those who stayed (teal) and those who left (clay). "
                       "The number at the end of each bar is the department's headcount.",
                  why="Research & Development holds most of the workforce, so it contributes the most leavers in "
                      "absolute terms even where its rate is lower.",
                  key="ov_dept", on_select="rerun", selection_mode="points")
    pts = (ev.selection.points if ev is not None and hasattr(ev, "selection") else []) or []
    if pts:
        picked = pts[0].get("y")
        if picked and picked != drill:
            st.session_state["ov_drill"] = picked
            st.rerun()
    if drill and st.button("Reset drill-down", icon=":material/undo:", key="ov_reset"):
        st.session_state.pop("ov_drill", None)
        st.rerun()
    r_dept = rate_by(df_all if drill else df, "Department")
    ui.chart(C.rate_dot_chart(r_dept, k["attrition_rate"] if not drill else overall, height=210,
                              highlight=drill), "Attrition rate by department",
             "95% Wilson interval" + (" · drilled department highlighted" if drill else ""), key="ov_dept_rate",
             what="Share of each department's employees who left, with its 95% interval.",
             why="Rates, not headcounts, show where leaving is disproportionate.")
with right:
    with ui.card("Stayed vs left", "Share of the current selection", key="ov_split"):
        st.plotly_chart(C.donut_free_split(k["stayed"], k["left"]), width="stretch", theme=None,
                        config={"displayModeBar": False}, alt="Stayed versus left split")
        st.caption(f"{k['left']:,} of {k['headcount']:,} employees left. The full dataset rate is {overall:.1%}; "
                   "IBM's sample is imbalanced at roughly 5 stayers per leaver.")
    with ui.card("Largest attrition gaps", "Groups whose interval sits clear of everyone else's", key="ov_find"):
        ui.findings(headline_findings(df, k=4, min_n=30 if len(df) > 300 else 15))

ui.section("Where attrition concentrates", "rate per group, 95% Wilson interval, hollow dots mark n < 30")
c1, c2 = st.columns(2, gap="medium")
with c1:
    r = rate_by(df, "JobRole")
    ui.chart(C.rate_dot_chart(r, k["attrition_rate"]), "Attrition by job role",
             what="Each dot is the share of a role's employees who left; the whisker is the 95% Wilson interval. "
                  "The vertical rule is the rate for the current selection.",
             why="Roles with intervals entirely to the right of the rule leave more often than the average employee in "
                 "this sample. Wide whiskers mean small groups, where a few people change the rate a lot.",
             key="ov_role")
with c2:
    r = rate_by(df, "OverTime")
    r["group"] = r["group"].map({"Yes": "Works overtime", "No": "No overtime"})
    ui.chart(C.rate_dot_chart(r, k["attrition_rate"], height=200), "Attrition by overtime", key="ov_ot",
             what="Attrition among employees who work overtime against those who do not.",
             why="Overtime is the single categorical attribute with the largest gap in this dataset. It is an "
                 "association: overtime may stand in for workload, role or seniority.")
    r = rate_by(df, "AgeBand")
    ui.chart(C.rate_dot_chart(r, k["attrition_rate"], ordered=True, height=250), "Attrition by age band",
             key="ov_age",
             what="Attrition by age band, youngest at the top.",
             why="Younger employees leave more often here, which overlaps with lower job levels and shorter tenure.")

ui.section("Model snapshot", "read from the saved experiment, never typed in")
b = bundle()
if b is None:
    ui.empty_state("No trained model yet", "Run `python scripts/train_default.py` or use the Model lab.",
                   ":material/model_training:")
else:
    tm = b.test_metrics
    bt = b.test_bootstrap.set_index("metric") if not b.test_bootstrap.empty else None

    def ci(m):
        return f"95% CI {bt.loc[m, 'ci_low']:.2f} to {bt.loc[m, 'ci_high']:.2f}" if bt is not None else ""

    ui.tiles([
        ("Selected model", b.selection["model_name"], b.selection["strategy_name"], T.BLUE),
        ("Held-out PR-AUC", f"{tm['pr_auc']:.3f}", ci("pr_auc") + f" · chance {b.metadata['split']['test_positives'] / b.metadata['split']['test_rows']:.3f}", T.TEAL),
        ("Held-out ROC-AUC", f"{tm['roc_auc']:.3f}", ci("roc_auc"), T.TEAL),
        ("Recall at threshold", f"{tm['recall']:.0%}", f"threshold {b.threshold:.2f} · precision {tm['precision']:.0%}", T.CLAY),
    ])
    st.caption(f"Test set: {b.metadata['split']['test_rows']} employees held out before any fitting and scored once. "
               "Open Performance for cross-validated comparisons and the full evaluation.")
