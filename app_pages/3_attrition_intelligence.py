"""Attrition intelligence: where and among whom leaving concentrates."""
import streamlit as st

from talentscope.analytics import (
    DIMENSIONS,
    association_ranking,
    crosstab_rates,
    headline_findings,
    numeric_comparison,
    rate_by,
    workforce_kpis,
)
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import dataset, filter_bar, filter_summary

df_all = dataset()
ui.page_header(
    "03 · Understand", "Attrition intelligence",
    "Attrition rates across ten workforce dimensions, each with its denominator and a 95% Wilson interval, "
    "then a significance-tested ranking of every attribute. These are associations in a synthetic sample; "
    "none of them shows that changing an attribute would change whether someone leaves.")
df = filter_bar(df_all)
st.caption(filter_summary(df_all, df))
if len(df) < 20:
    ui.empty_state("Too few employees for rates", "Rates on fewer than 20 employees are not shown. Widen the filters.",
                   ":material/filter_alt_off:")
    st.stop()

k = workforce_kpis(df)
lo, hi = k["rate_ci"]
assoc = association_ranking(df)
sig = int(assoc["significant"].sum()) if not assoc.empty else 0
top = assoc.iloc[0] if not assoc.empty else None
ui.tiles([
    ("Attrition rate", f"{k['attrition_rate']:.1%}", f"95% CI {lo:.1%} to {hi:.1%} · n = {k['headcount']:,}", T.CLAY),
    ("Leavers", f"{k['left']:,}", f"of {k['headcount']:,} employees", T.CLAY),
    ("Significant attributes", f"{sig} / {len(assoc)}", "BH-adjusted p < 0.05", T.BLUE),
    ("Strongest association", top["label"] if top is not None else "n/a",
     f"{top['effect_name']} = {top['effect']:.2f}" if top is not None else "", T.TEAL),
])

ui.section("Headline gaps", "groups whose interval sits clear of the rest of the selection")
with ui.card(key="ai_find"):
    ui.findings(headline_findings(df, k=5, min_n=30 if len(df) > 300 else 15))

ui.section("Attrition across the workforce", "hollow dots: fewer than 30 employees, read with caution")
min_n = st.slider("Hide groups smaller than", 1, 50, 5, key="ai_min_n",
                  help="Groups below this size are removed from the charts. Their rates are too unstable to compare.")
overall = k["attrition_rate"]

GRID = [
    ("department", "Department", "Department sets the workforce context; Sales loses a larger share than R&D."),
    ("job_role", "Job role", "Sales Representatives and Laboratory Technicians have the highest rates; directors and managers the lowest."),
    ("overtime", "Overtime", "Overtime workers leave at roughly three times the rate of others in the full dataset."),
    ("age", "Age band", "Attrition is highest among employees under 26 and lowest between 36 and 45; the 56+ group is small."),
    ("income", "Monthly income band", "The lowest income band has the highest rate; income tracks job level closely."),
    ("tenure", "Years at company", "Early tenure is the riskiest period: most leavers go in their first years."),
    ("job_satisfaction", "Job satisfaction", "Low satisfaction carries a higher rate, but the gradient across the other three levels is shallow."),
    ("work_life", "Work-life balance", "'Bad' work-life balance has the highest rate, but only 80 employees report it, so its interval is wide."),
    ("travel", "Business travel", "Frequent travellers leave more often than non-travellers."),
    ("promotion", "Last promotion", "Employees promoted this year leave about as often as those last promoted six or more years ago; the intervals overlap, so recency of promotion shows no clear pattern."),
]
for i in range(0, len(GRID), 2):
    cols = st.columns(2, gap="medium")
    for col, (dim, title, why) in zip(cols, GRID[i:i + 2]):
        column, label, ordered = DIMENSIONS[dim]
        r = rate_by(df, column, min_n=min_n)
        if dim == "overtime":
            r["group"] = r["group"].map({"Yes": "Works overtime", "No": "No overtime"})
        if dim == "travel":
            r["group"] = r["group"].str.replace("_", " ")
        with col:
            if r.empty:
                ui.empty_state(title, "No group meets the minimum size.")
                continue
            ui.chart(C.rate_dot_chart(r, overall, ordered=ordered), f"Attrition by {label.lower()}", key=f"ai_{dim}",
                     what=f"Share of each {label.lower()} group who left, with the 95% Wilson interval. The rule marks the "
                          f"selection's overall rate ({overall:.1%}).",
                     why=why + " Observed in the full dataset; your filters may change it.")

ui.section("Two attributes at once", "attrition rate in every cell; cells with fewer than the minimum size are blank")
dim_names = {k_: v[1] for k_, v in DIMENSIONS.items()}
c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
row_dim = c1.selectbox("Rows", list(dim_names), index=list(dim_names).index("job_role"), format_func=dim_names.get,
                       key="ai_rows")
col_dim = c2.selectbox("Columns", list(dim_names), index=list(dim_names).index("overtime"), format_func=dim_names.get,
                       key="ai_cols")
if row_dim == col_dim:
    c3.caption("Pick two different attributes.")
else:
    rates, counts = crosstab_rates(df, DIMENSIONS[row_dim][0], DIMENSIONS[col_dim][0], min_n=max(min_n, 10))
    ui.chart(C.heatmap(rates, colorscale=T.SEQ_CLAY, zmin=0, zmax=max(0.5, float(rates.max().max() or 0)),
                       counts=counts, colorbar_title="attrition", hover_label="attrition"),
             f"{dim_names[row_dim]} x {dim_names[col_dim]}", key="ai_cross",
             what="Attrition rate for each combination; hover for the cell size. Blank cells have too few employees.",
             why="Interactions show whether one gap holds within the other attribute's groups, for example whether "
                 "overtime matters in every role or only some.")

ui.section("Every attribute, ranked", "chi-square for categorical, Mann-Whitney U for continuous, Benjamini-Hochberg adjusted")
c1, c2 = st.columns([1.1, 1], gap="medium")
with c1:
    a = assoc.head(20)
    ui.chart(C.ranked_bars(a["label"], a["effect"], color=T.CLAY, fmt=".2f", xlabel="effect size",
                           highlight=set(a.loc[a["significant"], "label"]),
                           hover_extra=[f"{t} · {e} · adj. p = {p:.2g}" + (f" · {d}" if d else "")
                                        for t, e, p, d in zip(a["test"], a["effect_name"], a["p_adjusted"], a["direction"])]),
             "Association with attrition", "gray bars are not significant after correction", key="ai_assoc",
             what="Cramer's V for categorical and ordinal attributes; the absolute rank-biserial correlation for "
                  "continuous ones. Both run from 0 (no association) to 1.",
             why="Effects are small to moderate. No single attribute separates leavers from stayers, which is why "
                 "the model combines them.")
with c2:
    with ui.card("Test results", key="ai_tbl"):
        show = assoc[["label", "test", "effect", "direction", "p_adjusted", "significant"]].rename(columns={
            "label": "Attribute", "test": "Test", "effect": "Effect", "direction": "Direction",
            "p_adjusted": "Adj. p", "significant": "Sig."})
        st.dataframe(show, hide_index=True, height=520, alt="Association test table",
                     column_config={"Effect": st.column_config.ProgressColumn("Effect", format="%.2f", min_value=0,
                                                                              max_value=max(0.4, float(show["Effect"].max()))),
                                    "Adj. p": st.column_config.NumberColumn(format="%.1e")})

ui.section("Leavers against stayers", "continuous attributes")
nc = numeric_comparison(df)
if not nc.empty:
    nc = nc.sort_values("rank_biserial")
    show = nc[["label", "median_left", "median_stayed", "mean_left", "mean_stayed", "rank_biserial", "p_adjusted"]]
    st.dataframe(show.rename(columns={"label": "Attribute", "median_left": "Median (left)", "median_stayed": "Median (stayed)",
                                      "mean_left": "Mean (left)", "mean_stayed": "Mean (stayed)",
                                      "rank_biserial": "Rank-biserial r", "p_adjusted": "Adj. p"}),
                 hide_index=True, alt="Leaver versus stayer comparison",
                 column_config={"Adj. p": st.column_config.NumberColumn(format="%.1e"),
                                "Mean (left)": st.column_config.NumberColumn(format="%.1f"),
                                "Mean (stayed)": st.column_config.NumberColumn(format="%.1f"),
                                "Rank-biserial r": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("Negative r: leavers tend to have lower values. Tenure, experience, income and age all move together, "
               "so these are not independent findings.")
ui.note("<b>Reading these results.</b> The data are IBM's fictional sample of 1,470 employees, and every gap above is "
        "an association within it. Small groups have wide intervals, many comparisons are made at once (hence the "
        "Benjamini-Hochberg correction), and correlated attributes such as job level, income, age and tenure describe "
        "overlapping people. Treat these as prompts for investigation, not as levers.", T.AMBER)
