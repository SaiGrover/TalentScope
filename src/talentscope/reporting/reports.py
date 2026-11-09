"""Report builders.

Each builder returns Markdown assembled from computed tables; ``to_html``
wraps Markdown in a self-contained, print-friendly page. Nothing in a report
is typed by hand: every number comes from the frame or the bundle passed in.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from talentscope import __version__, config
from talentscope.analytics import DIMENSIONS, association_ranking, headline_findings, rate_by, workforce_kpis
from talentscope.data import schema


def _rel(path) -> str:
    """Repository-relative path for display (never a machine-specific absolute path)."""
    from pathlib import Path
    try:
        return Path(path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt.get(c, ".3f")) if not np.isnan(v) else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _header(title: str, filters_note: str = "") -> str:
    return (f"# {title}\n\nTalentScope {__version__} · generated {_stamp()}  \n"
            f"Dataset: {config.DATASET_NAME} (fictional data created by IBM; not real employees)."
            + (f"  \nSelection: {filters_note}" if filters_note else "") + "\n")


def _rate_table(df: pd.DataFrame, col: str, label: str) -> str:
    r = rate_by(df, col)
    r = r[["group", "n", "left", "rate", "ci_low", "ci_high", "lift"]].rename(columns={
        "group": label, "n": "Employees", "left": "Left", "rate": "Attrition", "ci_low": "95% CI low",
        "ci_high": "95% CI high", "lift": "x overall"})
    return _md_table(r, {"Attrition": ".1%", "95% CI low": ".1%", "95% CI high": ".1%", "x overall": ".2f"})


def workforce_report(df: pd.DataFrame, filters_note: str = "") -> str:
    k = workforce_kpis(df)
    lo, hi = k["rate_ci"]
    parts = [_header("Workforce analytics report", filters_note), "## Headline figures\n",
             _md_table(pd.DataFrame([
                 ("Employees", f"{k['headcount']:,}"), ("Left", f"{k['left']:,}"), ("Stayed", f"{k['stayed']:,}"),
                 ("Attrition rate", f"{k['attrition_rate']:.1%} (95% CI {lo:.1%} to {hi:.1%})"),
                 ("Mean age", f"{k['avg_age']:.1f}"), ("Mean / median tenure", f"{k['avg_tenure']:.1f} / {k['median_tenure']:.0f} years"),
                 ("Mean / median monthly income", f"{k['avg_income']:,.0f} / {k['median_income']:,.0f}"),
                 ("Overtime share", f"{k['overtime_share']:.1%}")], columns=["Measure", "Value"])), ""]
    for col, label in (("Department", "Department"), ("JobRole", "Job role"), ("JobLevelLabel", "Job level"),
                       ("AgeBand", "Age band"), ("Gender", "Gender"), ("MaritalStatus", "Marital status"),
                       ("EducationLabel", "Education")):
        g = df.groupby(col, observed=True).agg(Employees=("Age", "size"), MeanAge=("Age", "mean"),
                                               MedianIncome=("MonthlyIncome", "median"),
                                               MeanTenure=("YearsAtCompany", "mean"),
                                               Overtime=("OverTime", lambda s: (s == "Yes").mean())).reset_index()
        g = g.rename(columns={col: label, "MeanAge": "Mean age", "MedianIncome": "Median income",
                              "MeanTenure": "Mean tenure (yrs)", "Overtime": "Overtime share"})
        g["Share"] = g["Employees"] / len(df)
        parts += [f"## {label}\n", _md_table(g, {"Mean age": ".1f", "Median income": ",.0f", "Mean tenure (yrs)": ".1f",
                                                 "Overtime share": ".0%", "Share": ".1%"}), ""]
    sat = []
    for col in ("JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement", "WorkLifeBalance"):
        vc = df[col].value_counts(normalize=True).sort_index()
        sat.append({"Survey item": schema.PRETTY_NAMES[col],
                    **{schema.ORDINAL_LEVELS[col][i]: vc.get(i, 0.0) for i in schema.ORDINAL_LEVELS[col]}})
    parts += ["## Engagement survey (share of employees per level)\n",
              _md_table(pd.DataFrame(sat).fillna(0.0), {c: ".0%" for c in ["Low", "Medium", "High", "Very High", "Bad",
                                                                           "Good", "Better", "Best"]}), ""]
    return "\n".join(parts)


def attrition_report(df: pd.DataFrame, filters_note: str = "") -> str:
    k = workforce_kpis(df)
    lo, hi = k["rate_ci"]
    parts = [_header("Attrition analysis report", filters_note),
             f"Overall attrition: **{k['attrition_rate']:.1%}** ({k['left']:,} of {k['headcount']:,}; 95% Wilson CI "
             f"{lo:.1%} to {hi:.1%}).\n",
             "All rates below are descriptive associations within a synthetic sample. Intervals are Wilson 95% "
             "intervals; small groups have wide intervals and should not be over-read.\n",
             "## Largest gaps\n"]
    for f in headline_findings(df, k=6, min_n=30 if len(df) > 300 else 15):
        parts.append(f"- {f['text']} ({f['ratio']:.1f}x)")
    parts.append("")
    for key in ("department", "job_role", "overtime", "age", "income", "tenure", "job_satisfaction", "work_life",
                "travel", "promotion", "marital", "stock"):
        col, label, _ = DIMENSIONS[key]
        parts += [f"## Attrition by {label.lower()}\n", _rate_table(df, col, label), ""]
    a = association_ranking(df)
    if not a.empty:
        a = a[["label", "test", "effect_name", "effect", "direction", "p_value", "p_adjusted", "significant"]].rename(
            columns={"label": "Attribute", "test": "Test", "effect_name": "Effect size", "effect": "Value",
                     "direction": "Direction", "p_value": "p", "p_adjusted": "BH-adjusted p", "significant": "Significant"})
        parts += ["## Association tests (Benjamini-Hochberg adjusted)\n",
                  _md_table(a, {"Value": ".3f", "p": ".2e", "BH-adjusted p": ".2e"}), ""]
    return "\n".join(parts)


def model_evaluation_report(b, explanations: dict | None = None) -> str:
    sel, tm, md = b.selection, b.test_metrics, b.metadata
    bt = b.test_bootstrap
    cv = b.cv_summary.sort_values("cv_pr_auc_mean", ascending=False)
    parts = [_header("Model evaluation summary"),
             f"**Selected configuration:** {sel['model_name']} with {sel['strategy_name'].lower()}  \n"
             f"**Selection rule:** {sel['reason']}  \n"
             f"**Decision threshold:** {sel['threshold']:.2f}, maximising {sel['threshold_metric'].upper()} on "
             f"out-of-fold training predictions.  \n"
             f"**Run created:** {md['created_utc']} · runtime {md['runtime_seconds']:.0f} s · seed {b.config['seed']}\n",
             "## Protocol\n"] + [f"- **{k}**: {v}" for k, v in md["protocol"].items()] + [
             "", f"Split: {md['split']['train_rows']} training employees ({md['split']['train_positives']} leavers), "
                 f"{md['split']['test_rows']} held-out employees ({md['split']['test_positives']} leavers).\n",
             "## Held-out test results (scored once)\n",
             _md_table(bt[["label", "estimate", "ci_low", "ci_high"]].rename(columns={
                 "label": "Metric", "estimate": "Estimate", "ci_low": "95% CI low", "ci_high": "95% CI high"})),
             "", f"Confusion matrix at {sel['threshold']:.2f}: TP {tm['tp']}, FP {tm['fp']}, TN {tm['tn']}, FN {tm['fn']}.\n",
             "## Cross-validated comparison (training split)\n",
             _md_table(cv[["model_name", "strategy_name", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
                           "cv_brier_mean", "oof_threshold", "oof_f1", "selected"]].rename(columns={
                 "model_name": "Algorithm", "strategy_name": "Strategy", "cv_pr_auc_mean": "PR-AUC",
                 "cv_pr_auc_std": "PR-AUC SD", "cv_roc_auc_mean": "ROC-AUC", "cv_brier_mean": "Brier",
                 "oof_threshold": "OOF threshold", "oof_f1": "OOF F1", "selected": "Selected"}),
                 {"OOF threshold": ".2f"}), ""]
    none = cv[cv["strategy"] == "none"].set_index("model")
    sm = cv[cv["strategy"] == "smotenc"].set_index("model")
    both = none.index.intersection(sm.index)
    if len(both):
        d = (sm.loc[both, "cv_pr_auc_mean"] - none.loc[both, "cv_pr_auc_mean"])
        parts += ["## SMOTENC against no rebalancing\n",
                  _md_table(pd.DataFrame({"Algorithm": none.loc[both, "model_name"].values,
                                          "No rebalancing": none.loc[both, "cv_pr_auc_mean"].values,
                                          "SMOTENC": sm.loc[both, "cv_pr_auc_mean"].values, "Change": d.values}),
                            {"Change": "+.3f"}),
                  f"\nSMOTENC raised mean CV PR-AUC for {int((d > 0).sum())} of {len(d)} algorithms in this run.\n"]
    if explanations:
        if "shap_importance_by_feature" in explanations:
            s = explanations["shap_importance_by_feature"].head(10).copy()
            s["feature"] = s["feature"].map(lambda f: schema.PRETTY_NAMES.get(f, f))
            parts += ["## Top attributes by mean |SHAP| (held-out set)\n", _md_table(s, {"mean_abs_shap": ".4f"}), ""]
        if "permutation_importance" in explanations:
            p = explanations["permutation_importance"].head(10)[["label", "importance_mean", "importance_std"]]
            parts += ["## Top attributes by permutation importance (PR-AUC drop, held-out set)\n",
                      _md_table(p, {"importance_mean": ".4f", "importance_std": ".4f"}), ""]
        if "group_metrics" in explanations:
            g = explanations["group_metrics"]
            parts += ["## Error rates by group (held-out set)\n",
                      _md_table(g, {c: ".3f" for c in g.columns if g[c].dtype.kind == "f"}), ""]
    parts += ["## Limitations\n",
              "- Fictional data from IBM; 1,470 employees and 237 leavers. Held-out estimates rest on about 47 leavers "
              "and carry wide intervals.",
              "- Associations only; SHAP and permutation importance explain the model, not the causes of attrition.",
              "- The model uses gender, age and marital status because the dataset includes them. Real use would require "
              "a review of those inputs and a fairness assessment on real outcomes.", ""]
    return "\n".join(parts)


def model_card(b, explanations: dict | None = None) -> str:
    sel, tm, md = b.selection, b.test_metrics, b.metadata
    bt = b.test_bootstrap.set_index("metric")
    top = ""
    if explanations and "shap_importance_by_feature" in explanations:
        feats = explanations["shap_importance_by_feature"]["feature"].head(5)
        top = ", ".join(schema.PRETTY_NAMES.get(f, f) for f in feats)
    v = md["versions"]
    return "\n".join([
        f"# Model card: TalentScope attrition model ({sel['model_name']}, {sel['strategy_name'].lower()})\n",
        f"Generated {_stamp()} from `{_rel(b.run_dir)}`.\n",
        "## Model details",
        f"- Pipeline: median/mode imputation → {'resampling (' + sel['strategy_name'] + ') → ' if sel['strategy'] in ('smotenc', 'random_oversample') else ''}"
        f"integer rounding{', standardisation' if sel['model'] in ('logreg', 'svm') else ''} → one-hot encoding → {sel['model_name']}",
        f"- Tuned hyperparameters: `{sel['params']}`",
        f"- Decision threshold: {sel['threshold']:.2f} ({sel['threshold_source']})",
        f"- Random seed: {b.config['seed']}; libraries: " + ", ".join(f"{k} {val}" for k, val in v.items()),
        f"- Artefact: `{_rel(b.model_path)}` (joblib)\n",
        "## Intended use",
        "- Portfolio demonstration of leakage-safe imbalanced classification and explainability on a public synthetic "
        "dataset.",
        "- Not intended for decisions about real employees: hiring, promotion, pay, performance management or "
        "dismissal.\n",
        "## Training data",
        f"- {md['dataset']['name']}, {md['dataset']['rows']:,} rows, sha256 `{md['dataset']['sha256']}`.",
        f"- Source: {md['dataset']['source_url']} (copy taken from {md['dataset']['mirror_url']}).",
        "- Fictional data created by IBM data scientists. Identifier and constant columns are removed; 30 attributes "
        "are used.",
        f"- Stratified split: {md['split']['train_rows']} training / {md['split']['test_rows']} held-out employees.\n",
        "## Evaluation (held-out set, scored once, 95% stratified bootstrap intervals)",
        *[f"- {bt.loc[m, 'label']}: {bt.loc[m, 'estimate']:.3f} ({bt.loc[m, 'ci_low']:.3f} to {bt.loc[m, 'ci_high']:.3f})"
          for m in ("pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "brier")],
        f"- Cross-validated PR-AUC on the training split: {sel['cv_pr_auc_mean']:.3f} ± {sel['cv_pr_auc_std']:.3f} (SD over folds).",
        f"- Base rate (chance PR-AUC): {md['split']['test_positives'] / md['split']['test_rows']:.3f}.\n",
        "## Explainability",
        f"- Most influential attributes by mean |SHAP|: {top}." if top else "- See the Explainability page.",
        "- SHAP values describe the model's use of each input, not causal effects.\n",
        "## Ethical considerations",
        "- Inputs include gender, age and marital status. Error rates by group are reported in the evaluation summary; "
        "differences there come from small groups and from patterns in synthetic data.",
        "- A risk score must not be used to treat employees differently. If used at all, it should prompt a supportive "
        "conversation reviewed by a person.\n",
        "## Caveats",
        "- Small sample: 237 leavers in total. Expect the held-out metrics to move by several points on another split.",
        "- The data is synthetic, so the patterns may not transfer to any real organisation.",
    ]) + "\n"


def experiment_comparison_csv(b) -> bytes:
    cols = ["model_name", "strategy_name", "n_folds", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
            "cv_roc_auc_std", "cv_brier_mean", "cv_f1_mean", "cv_recall_mean", "cv_precision_mean",
            "cv_balanced_accuracy_mean", "oof_threshold", "oof_f1", "oof_precision", "oof_recall", "fit_seconds", "selected"]
    return b.cv_summary[[c for c in cols if c in b.cv_summary.columns]].to_csv(index=False).encode("utf-8")


def predictions_csv(b, X_test: pd.DataFrame) -> bytes:
    tp = b.test_predictions.copy()
    tp["risk_band"] = np.where(tp["probability"] >= b.threshold, "high",
                               np.where(tp["probability"] >= config.RISK_LOW_FRACTION * b.threshold, "elevated", "low"))
    feats = X_test.loc[tp["row_id"], ["Department", "JobRole", "JobLevel", "OverTime", "YearsAtCompany"]].reset_index(drop=True)
    out = pd.concat([tp.reset_index(drop=True), feats], axis=1)
    out.insert(0, "note", "synthetic IBM sample; held-out demonstration only")
    return out.to_csv(index=False).encode("utf-8")


_CSS = """
body{font-family:Inter,Segoe UI,system-ui,sans-serif;max-width:980px;margin:2.5rem auto;padding:0 1.2rem;color:#1d2a31;
line-height:1.55;font-size:14.5px}h1{font-size:1.7rem;letter-spacing:-.02em;border-bottom:3px solid #2fa697;
padding-bottom:.4rem}h2{font-size:1.15rem;margin-top:2rem;color:#13353a}table{border-collapse:collapse;width:100%;
margin:.6rem 0;font-size:13px}th,td{border:1px solid #d5dee2;padding:.32rem .5rem;text-align:left}
th{background:#eef4f5}tr:nth-child(even) td{background:#fafcfc}code{background:#eef4f5;padding:.05rem .3rem;
border-radius:4px}li{margin:.15rem 0}@media print{body{margin:0}}
"""


def to_html(markdown_text: str, title: str) -> str:
    """Minimal Markdown -> HTML (headings, lists, tables, bold, code) with no extra dependency."""
    lines = markdown_text.splitlines()
    out, i = [], 0

    def inline(s: str) -> str:
        s = html.escape(s)
        import re
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s.replace("  ", "&nbsp; ") if s.endswith("  ") else s

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            head = [c.strip() for c in ln.strip("|").split("|")]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue
        if ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        if ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.strip():
            out.append(f"<p>{inline(ln.rstrip())}</p>")
        i += 1
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style>"
            f"</head><body>{''.join(out)}</body></html>")
