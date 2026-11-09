"""Workforce explorer: who works here."""
import plotly.graph_objects as go
import streamlit as st

from talentscope.analytics import workforce_kpis
from talentscope.data import schema
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import dataset, filter_bar, filter_summary

df_all = dataset()
ui.page_header(
    "02 · Understand", "Workforce explorer",
    "Demographics, pay, careers and engagement for any slice of the workforce. Filters here also apply on "
    "Overview, Attrition and Data lab.")
df = filter_bar(df_all, extended=True)
st.caption(filter_summary(df_all, df))
if df.empty:
    ui.empty_state("No employees match these filters", "Clear a filter above to see results.", ":material/filter_alt_off:")
    st.stop()

k = workforce_kpis(df)
ui.tiles([
    ("Employees", f"{k['headcount']:,}", f"{k['headcount'] / len(df_all):.0%} of the workforce", T.BLUE),
    ("Mean age", f"{k['avg_age']:.1f}", f"range {df['Age'].min()} to {df['Age'].max()}", T.TEAL),
    ("Median income", f"{k['median_income']:,.0f}", "per month, synthetic units", T.AMBER),
    ("Median tenure", f"{k['median_tenure']:.0f} yrs", f"mean {k['avg_tenure']:.1f} yrs", T.TEAL),
    ("Overtime", f"{k['overtime_share']:.0%}", "work overtime", T.VIOLET),
    ("Female share", f"{(df['Gender'] == 'Female').mean():.0%}", "as recorded", T.BLUE),
])

view = st.segmented_control("View", ["Organisation", "Demographics", "Pay & level", "Career", "Engagement", "Table"],
                            default="Organisation", key="wf_view", label_visibility="collapsed")

if view == "Organisation":
    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        ui.chart(C.stacked_outcome_bars(df, "Department"), "Department", key="wf_dept",
                 what="Headcount per department, split by outcome.",
                 why="Sets the denominators for every department-level rate elsewhere in the app.")
        tr = df["BusinessTravel"].value_counts().reindex(["Non-Travel", "Travel_Rarely", "Travel_Frequently"]).fillna(0)
        ui.chart(C.ranked_bars([s.replace("_", " ") for s in tr.index], tr.values, fmt=",.0f", color=T.BLUE,
                               xlabel="employees", height=200), "Business travel", key="wf_travel")
    with c2:
        ui.chart(C.stacked_outcome_bars(df, "JobRole"), "Job role", key="wf_role",
                 what="Headcount per job role, split by outcome, largest role at the top.",
                 why="Sales Executives and Research Scientists are the two largest roles; small roles such as Human "
                     "Resources need wide intervals on any rate.")

elif view == "Demographics":
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.chart(C.histogram_by_outcome(df, "Age", nbins=22, xlabel="age"), "Age distribution", key="wf_age",
                 what="Age distribution of stayers and leavers, each normalised to 100% so the shapes compare.",
                 why="Leavers skew younger: the clay distribution peaks in the late twenties.")
    with c2:
        ui.chart(C.stacked_outcome_bars(df, "MaritalStatus"), "Marital status", key="wf_marital", height=230)
        ui.chart(C.stacked_outcome_bars(df, "EducationLabel", order=[l for l in schema.ORDINAL_LEVELS["Education"].values()
                                                                     if l in set(df["EducationLabel"].astype(str))]),
                 "Education level", key="wf_edu", height=260)
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        ui.chart(C.stacked_outcome_bars(df, "EducationField"), "Education field", key="wf_field", height=280)
    with c4:
        ui.chart(C.stacked_outcome_bars(df, "Gender"), "Gender", key="wf_gender", height=200)
        ui.chart(C.stacked_outcome_bars(df, "DistanceBand", order=[b for b in ["1-5", "6-10", "11-20", "21-29"]
                                                                   if b in set(df["DistanceBand"].astype(str))]),
                 "Commute distance", key="wf_dist", height=240)

elif view == "Pay & level":
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.chart(C.histogram_by_outcome(df, "MonthlyIncome", nbins=30, xlabel="monthly income"), "Monthly income",
                 key="wf_inc", what="Income distribution of stayers and leavers, normalised within each group.",
                 why="In the full dataset 58% of leavers earn under 4,000 a month, against 33% of stayers.")
    with c2:
        ui.chart(C.box_by(df, "JobLevel", "MonthlyIncome", ylabel="monthly income", order=[1, 2, 3, 4, 5]),
                 "Income by job level", key="wf_inclvl",
                 what="Income spread at each job level, stayers and leavers side by side.",
                 why="Income and job level correlate at r = 0.95, so their effects on attrition are hard to separate.")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        ct = df.pivot_table(index="EducationLabel", columns="JobLevelLabel", values="AttritionFlag", aggfunc="size",
                            observed=True).fillna(0)
        ui.chart(C.heatmap(ct, colorscale=T.SEQ_TEAL, text_fmt=",.0f", colorbar_title="employees",
                           hover_label="employees"), "Education x job level", key="wf_edulvl",
                 what="Headcount for every combination of education and job level.",
                 why="Most employees sit at levels 1 and 2 regardless of education.")
    with c4:
        ui.chart(C.histogram_by_outcome(df, "PercentSalaryHike", nbins=15, xlabel="last salary hike, %"),
                 "Salary hike", key="wf_hike")

elif view == "Career":
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.chart(C.histogram_by_outcome(df, "YearsAtCompany", nbins=20, xlabel="years at company"), "Tenure",
                 key="wf_ten", what="Years at the company, normalised within each outcome group.",
                 why="In the full dataset 43% of leavers had two years or less at the company, against 20% of stayers.")
        ui.chart(C.histogram_by_outcome(df, "NumCompaniesWorked", nbins=10, xlabel="companies worked for"),
                 "Previous employers", key="wf_ncomp")
    with c2:
        ui.chart(C.histogram_by_outcome(df, "TotalWorkingYears", nbins=20, xlabel="total working years"),
                 "Total experience", key="wf_exp")
        ui.chart(C.histogram_by_outcome(df, "YearsSinceLastPromotion", nbins=15, xlabel="years since last promotion"),
                 "Years since promotion", key="wf_promo")

elif view == "Engagement":
    dims = {"JobSatisfactionLabel": "Job satisfaction", "EnvironmentSatisfactionLabel": "Environment",
            "RelationshipSatisfactionLabel": "Relationships", "JobInvolvementLabel": "Job involvement"}
    ui.chart(C.share_100(df, dims, ["Low", "Medium", "High", "Very High"], T.TEAL_RAMP[::-1]),
             "Survey ratings", "share of employees at each level, low to very high", key="wf_sat",
             what="Each bar splits employees across the four survey levels, darkest = low.",
             why="About a fifth of employees rate job, environment and relationship satisfaction as low, but only about "
                 "one in twenty rates their job involvement low. Attrition shows how these groups' rates compare.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.chart(C.share_100(df, {"WorkLifeBalanceLabel": "Work-life balance"}, ["Bad", "Good", "Better", "Best"],
                             T.TEAL_RAMP[::-1], height=170), "Work-life balance", key="wf_wlb")
        ot = df.groupby("Department")["OverTime"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
        ui.chart(C.ranked_bars(ot.index, ot.values, fmt=".0%", color=T.VIOLET, height=200), "Overtime share by department",
                 key="wf_ot")
    with c2:
        tt = df["TrainingTimesLastYear"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=tt.index, y=tt.values, marker=dict(color=T.BLUE, cornerradius=4),
                               hovertemplate="%{x} trainings: %{y} employees<extra></extra>"))
        fig.update_xaxes(title="trainings last year", dtick=1)
        fig.update_yaxes(title="employees")
        ui.chart(C.style(fig, 300), "Training last year", key="wf_train")

else:
    q = st.text_input("Search", placeholder="Search any column, e.g. Sales Executive, Single, Travel_Frequently",
                      key="wf_q", label_visibility="collapsed", icon=":material/search:")
    cols = ["Attrition", "Age", "Gender", "MaritalStatus", "Department", "JobRole", "JobLevel", "MonthlyIncome",
            "OverTime", "BusinessTravel", "YearsAtCompany", "TotalWorkingYears", "YearsSinceLastPromotion",
            "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "DistanceFromHome", "Education",
            "EducationField", "StockOptionLevel", "PercentSalaryHike", "PerformanceRating"]
    table = df[cols].copy()
    if q:
        mask = table.astype(str).apply(lambda c: c.str.contains(q, case=False, regex=False)).any(axis=1)
        table = table[mask]
    st.caption(f"{len(table):,} rows · synthetic records from the IBM sample")
    st.dataframe(table, hide_index=True, height=520, alt="Employee table",
                 column_config={"MonthlyIncome": st.column_config.NumberColumn("Monthly income", format="%d"),
                                "Attrition": st.column_config.TextColumn("Left?")})
    st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"), "talentscope_workforce.csv",
                       "text/csv", icon=":material/download:", key="wf_dl")
