"""Data laboratory: profiling, validation and free-form exploration."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from talentscope.analytics import correlation_matrix, cramers_v_matrix, outlier_table
from talentscope.data import schema
from talentscope.ui import charts as C
from talentscope.ui import components as ui
from talentscope.ui import tokens as T
from talentscope.ui.state import dataset, filter_bar, filter_summary, raw_dataset, validation_report

df_all = dataset()
raw = raw_dataset()
rep = validation_report()
ui.page_header(
    "04 · Understand", "Data laboratory",
    "The dataset as loaded: schema, a validation report, and tools for univariate, bivariate and correlation "
    "analysis. The validation report always describes the full file; the exploration tools follow the filters.")

ui.tiles([
    ("Rows x columns", f"{raw.shape[0]:,} x {raw.shape[1]}", "as shipped by IBM", T.BLUE),
    ("Checks passed", f"{rep.n_pass} / {len(rep.checks)}", f"{rep.n_warn} warnings · {rep.n_fail} failures",
     T.STATUS["pass"] if rep.n_fail == 0 else T.STATUS["fail"]),
    ("Missing cells", f"{int(rep.missing['missing'].sum()):,}", "across all columns", T.TEAL),
    ("Duplicates", f"{rep.duplicates}", "exact duplicate rows", T.TEAL),
    ("Dropped columns", f"{len(schema.DROP_COLUMNS)}", "identifier + 3 constants", T.AMBER),
    ("Model features", f"{len(schema.MODEL_FEATURES)}", f"{len(schema.NUMERIC_FEATURES)} numeric · {len(schema.ORDINAL_FEATURES)} ordinal · {len(schema.NOMINAL_FEATURES)} nominal", T.VIOLET),
])

tab = st.segmented_control("Section", ["Quality report", "Schema", "Preview", "Distributions", "Bivariate",
                                       "Correlations", "Outliers", "Target"], default="Quality report", key="dl_tab",
                           label_visibility="collapsed")

if tab == "Quality report":
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        with ui.card("Validation checks", "run on every load by talentscope.data.validate_dataset", key="dl_checks"):
            t = rep.to_frame()
            t["status"] = t["status"].map({"pass": "✓ pass", "warn": "! warn", "fail": "✕ fail"})
            st.dataframe(t, hide_index=True, height=560, alt="Validation checks",
                         column_config={"group": st.column_config.TextColumn("Group", width="small"),
                                        "name": st.column_config.TextColumn("Check"),
                                        "status": st.column_config.TextColumn("Status", width="small"),
                                        "detail": st.column_config.TextColumn("Detail", width="large")})
    with c2:
        ls = rep.leakage_screen.head(12)
        ui.chart(C.ranked_bars(ls["feature"].map(lambda f: schema.PRETTY_NAMES.get(f, f)), ls["univariate_auc"],
                               color=T.BLUE, fmt=".3f", xlabel="univariate ROC-AUC (0.5 = no signal)"),
                 "Leakage screen", "each feature alone, scored as a predictor of attrition", key="dl_leak",
                 what="ROC-AUC when one raw feature is used alone to rank employees by attrition. Categorical features "
                      "are encoded by their observed attrition rate for this screen only.",
                 why="A label proxy or post-outcome field would score near 1.0. The strongest single feature reaches "
                     f"{rep.leakage_screen['univariate_auc'].max():.2f}, so nothing here leaks the answer.")
        if rep.redundant_pairs is not None and len(rep.redundant_pairs):
            with ui.card("Correlated feature pairs", "|Pearson r| >= 0.7, kept with justification", key="dl_red"):
                rp = rep.redundant_pairs.copy()
                rp["feature_a"] = rp["feature_a"].map(lambda f: schema.PRETTY_NAMES.get(f, f))
                rp["feature_b"] = rp["feature_b"].map(lambda f: schema.PRETTY_NAMES.get(f, f))
                st.dataframe(rp, hide_index=True, alt="Correlated pairs")
    ui.note("<b>What was removed and why.</b> <code>EmployeeNumber</code> is a row identifier. <code>EmployeeCount</code>, "
            "<code>Over18</code> and <code>StandardHours</code> hold one value for every employee and carry no "
            "information. All four are dropped before modelling; the other 30 attributes are kept.", T.TEAL)

elif tab == "Schema":
    cp = rep.columns.copy()
    role = st.pills("Role", sorted(cp["role"].unique()), selection_mode="multi", key="dl_role")
    if role:
        cp = cp[cp["role"].isin(role)]
    st.dataframe(cp, hide_index=True, height=600, alt="Schema explorer",
                 column_config={"description": st.column_config.TextColumn("Description", width="large")})

elif tab == "Preview":
    n = st.slider("Rows", 5, 100, 20, key="dl_n")
    st.dataframe(raw.head(n), height=min(38 * n + 40, 620), alt="Raw data preview")
    st.download_button("Download the raw CSV", raw.to_csv(index=False).encode("utf-8"), "ibm_hr_employee_attrition.csv",
                       "text/csv", icon=":material/download:", key="dl_raw")

else:
    df = filter_bar(df_all)
    st.caption(filter_summary(df_all, df))
    if len(df) < 10:
        ui.empty_state("Too few employees", "Widen the filters to explore distributions.", ":material/filter_alt_off:")
        st.stop()
    numeric = schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES
    categorical = schema.NOMINAL_FEATURES + ["AgeBand", "IncomeBand", "TenureBand", "PromotionRecency"]
    label = lambda c: schema.PRETTY_NAMES.get(c, c)  # noqa: E731

    if tab == "Distributions":
        c1, c2 = st.columns([1, 2.4], gap="medium")
        with c1:
            feat = st.selectbox("Feature", numeric + schema.NOMINAL_FEATURES, format_func=label, key="dl_uni")
            split = st.toggle("Split by outcome", value=True, key="dl_split")
            s = df[feat]
            if feat in numeric:
                st.dataframe(s.describe().to_frame("value").T.round(2).T, alt="Summary statistics")
            else:
                st.dataframe(s.value_counts().rename("employees").to_frame(), alt="Category counts")
        with c2:
            if feat in numeric:
                if split:
                    fig = C.histogram_by_outcome(df, feat, nbins=min(30, int(s.nunique())), xlabel=label(feat), height=380)
                else:
                    fig = C.style(go.Figure(go.Histogram(x=s, nbinsx=min(30, int(s.nunique())), marker=dict(color=T.BLUE))),
                                  380)
                    fig.update_xaxes(title=label(feat)); fig.update_yaxes(title="employees")
                ui.chart(fig, f"{label(feat)}", key="dl_hist", what=schema.FEATURE_DESCRIPTIONS.get(feat, ""))
            else:
                fig = C.stacked_outcome_bars(df, feat) if split else C.ranked_bars(
                    s.value_counts().index, s.value_counts().values, fmt=",.0f", color=T.BLUE)
                ui.chart(fig, f"{label(feat)}", key="dl_bar", what=schema.FEATURE_DESCRIPTIONS.get(feat, ""))

    elif tab == "Bivariate":
        c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
        x = c1.selectbox("X", numeric + categorical, index=(numeric + categorical).index("TotalWorkingYears"),
                         format_func=label, key="dl_x")
        y = c2.selectbox("Y", numeric, index=numeric.index("MonthlyIncome"), format_func=label, key="dl_y")
        colour = c3.toggle("Colour by outcome", value=True, key="dl_col")
        if x in numeric:
            fig = go.Figure()
            groups = [("No", "Stayed", T.STAYED), ("Yes", "Left", T.LEFT)] if colour else [(None, "Employees", T.BLUE)]
            for key, name, c in groups:
                d = df if key is None else df[df["Attrition"] == key]
                fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", name=name,
                                           marker=dict(color=c, size=7, opacity=0.7, line=dict(color=T.PANEL, width=1)),
                                           hovertemplate=f"{label(x)} %{{x}}<br>{label(y)} %{{y:,}}<extra>{name}</extra>"))
            fig.update_xaxes(title=label(x)); fig.update_yaxes(title=label(y))
            rho = df[[x, y]].corr(method="spearman").iloc[0, 1]
            ui.chart(C.style(fig, 440), f"{label(y)} against {label(x)}", f"Spearman rho = {rho:.2f}", key="dl_scatter",
                     what="Each point is one employee.", why="Spearman's rho measures monotonic association; it is "
                     "robust to the skewed income and tenure distributions.")
        else:
            order = list(df[x].cat.categories) if hasattr(df[x], "cat") else sorted(df[x].astype(str).unique())
            ui.chart(C.box_by(df, x, y, ylabel=label(y), order=order, by_outcome=colour, height=420),
                     f"{label(y)} by {label(x)}", key="dl_box")

    elif tab == "Correlations":
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            cm = correlation_matrix(df)
            cm.index = [label(c) if c != "AttritionFlag" else "Attrition" for c in cm.index]
            cm.columns = cm.index
            ui.chart(C.heatmap(cm.round(2), colorscale=T.DIVERGING, zmin=-1, zmax=1, text_fmt=".1f", height=720,
                               colorbar_title="rho", hover_label="rho"),
                     "Spearman correlation", "numeric and ordinal features, plus the target", key="dl_corr",
                     what="Rank correlation between every pair of numeric and ordinal attributes. Blue is negative, "
                          "clay positive.",
                     why="A block of tenure, experience, age, level and income moves together; those features share "
                         "their signal in the model.")
        with c2:
            cv = cramers_v_matrix(df)
            cv.index = [label(c) for c in cv.index]
            cv.columns = cv.index
            ui.chart(C.heatmap(cv.round(2), colorscale=T.SEQ_TEAL, zmin=0, zmax=1, text_fmt=".2f", height=460,
                               colorbar_title="V", hover_label="Cramer's V"),
                     "Categorical association", "Cramer's V between nominal attributes and attrition", key="dl_cv",
                     what="Cramer's V for every pair of nominal attributes, including the target.",
                     why="Department and education field, and department and job role, are strongly linked by design: "
                         "roles belong to departments.")

    elif tab == "Outliers":
        ot = outlier_table(df)
        c1, c2 = st.columns([1, 1.3], gap="medium")
        with c1:
            ui.chart(C.ranked_bars(ot["label"], ot["outlier_pct"], fmt=".1%", color=T.AMBER, xlabel="share outside 1.5 x IQR"),
                     "IQR outliers", key="dl_out",
                     what="Share of employees outside Tukey's fences (1.5 x IQR beyond the quartiles).",
                     why="These values are plausible (senior earners, long-serving staff). Nothing is capped or removed; "
                         "tree models are insensitive to them and the linear models see standardised values.")
        with c2:
            feat = st.selectbox("Inspect", ot["feature"], format_func=label, key="dl_outf")
            ui.chart(C.box_by(df, "Department", feat, ylabel=label(feat), height=380), f"{label(feat)} by department",
                     key="dl_outbox")
        st.dataframe(ot.round(3), hide_index=True, alt="Outlier table")

    elif tab == "Target":
        k_ = df["Attrition"].value_counts()
        c1, c2 = st.columns([1, 1.4], gap="medium")
        with c1:
            with ui.card("Target distribution", key="dl_target"):
                st.plotly_chart(C.donut_free_split(int(k_.get("No", 0)), int(k_.get("Yes", 0))), width="stretch", theme=None,
                                config={"displayModeBar": False}, alt="Target split")
                st.dataframe(rep.target_distribution, hide_index=True, alt="Target counts")
                st.caption("The full file has 237 leavers and 1,233 stayers. With a 16% positive class, accuracy is "
                           "misleading (always predicting 'stay' scores 84%), so models are compared on PR-AUC.")
        with c2:
            mv = rep.missing
            ui.chart(C.ranked_bars(mv["column"], mv["missing_pct"], fmt=".1f", color=T.BLUE, xlabel="% missing",
                                   height=720), "Missing values per column", "the full file", key="dl_missing",
                     what="Percentage of missing cells in each of the 35 columns.",
                     why="There are none, but the pipeline still imputes (median / most frequent, fitted per training "
                         "fold) so it stays safe on new records with gaps.")
