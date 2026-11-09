"""Regenerate the results section of README.md from the committed default experiment.

    python scripts/update_readme_results.py

Everything between the RESULTS markers is rewritten from experiments/default,
so the README cannot drift from the evidence it cites.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from talentscope.explainability import load_explanations  # noqa: E402
from talentscope.modeling import load_bundle  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def build() -> str:
    b = load_bundle()
    sel, md = b.selection, b.metadata
    bt = b.test_bootstrap.set_index("metric")
    cv = b.cv_summary
    prev = md["split"]["test_positives"] / md["split"]["test_rows"]
    lines = [START, "",
             f"_Generated from `experiments/default` (run {md['created_utc']}, dataset sha256 "
             f"`{md['dataset']['sha256'][:12]}…`, seed {b.config['seed']}). Regenerate with "
             "`python scripts/train_default.py` then `python scripts/update_readme_results.py`._", "",
             f"**Selected:** {sel['model_name']} with {sel['strategy_name'].lower()}, C = {sel['params'].get('C', 'n/a'):.3g}, "
             f"decision threshold {sel['threshold']:.2f} (max F1 on out-of-fold training predictions).", "",
             f"**Held-out test set** ({md['split']['test_rows']} employees, {md['split']['test_positives']} leavers, scored once; "
             "95% stratified bootstrap intervals):", "",
             "| Metric | Estimate | 95% CI |", "|---|---|---|"]
    for m in ("pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "accuracy", "brier"):
        lines.append(f"| {bt.loc[m, 'label']} | {bt.loc[m, 'estimate']:.3f} | {bt.loc[m, 'ci_low']:.3f} to {bt.loc[m, 'ci_high']:.3f} |")
    lines += ["", f"Chance-level PR-AUC (the attrition rate) is {prev:.3f}. Always predicting the base rate gives a Brier "
                  f"score of {prev * (1 - prev):.3f}.", "",
              "**Cross-validated comparison** on the training split (3 × 5-fold repeated stratified CV, mean ± SD of PR-AUC):", "",
              "| Algorithm | No rebalancing | Class weights | SMOTENC | Random oversampling |", "|---|---|---|---|---|"]
    order = cv[cv["strategy"] == "none"].sort_values("cv_pr_auc_mean", ascending=False)["model"].tolist()
    for m in order:
        row = [cv[cv["model"] == m]["model_name"].iloc[0]]
        for s in ("none", "class_weight", "smotenc", "random_oversample"):
            r = cv[(cv["model"] == m) & (cv["strategy"] == s)]
            row.append("n/a" if r.empty else
                       (f"**{r['cv_pr_auc_mean'].iloc[0]:.3f} ± {r['cv_pr_auc_std'].iloc[0]:.3f}**" if r["selected"].iloc[0]
                        else f"{r['cv_pr_auc_mean'].iloc[0]:.3f} ± {r['cv_pr_auc_std'].iloc[0]:.3f}"))
        lines.append("| " + " | ".join(row) + " |")
    none = cv[cv["strategy"] == "none"].set_index("model")["cv_pr_auc_mean"]
    sm = cv[cv["strategy"] == "smotenc"].set_index("model")["cv_pr_auc_mean"]
    common = none.index.intersection(sm.index)
    d = (sm[common] - none[common])
    lines += ["", f"SMOTENC changed mean CV PR-AUC by {d.min():+.3f} to {d.max():+.3f} across {len(d)} algorithms "
                  f"(lower for {int((d < 0).sum())}). It raised recall at a 0.5 threshold, but at the cost of calibration, "
                  "and did not improve ranking quality. The experiment is reported as run; SMOTENC remains available "
                  "in the Model laboratory."]
    ex = load_explanations(b)
    if "shap_importance_by_feature" in ex and "permutation_importance" in ex:
        from talentscope.data import schema
        s = ex["shap_importance_by_feature"]["feature"].head(6).map(lambda f: schema.PRETTY_NAMES.get(f, f)).tolist()
        p = ex["permutation_importance"]["label"].head(6).tolist()
        lines += ["", f"**Most influential attributes** (held-out set). By mean |SHAP|: {', '.join(s)}. "
                      f"By permutation importance: {', '.join(p)}."]
    lines += ["", END]
    return "\n".join(lines)


def main() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    a, b = text.index(START), text.index(END) + len(END)
    path.write_text(text[:a] + build() + text[b:], encoding="utf-8")
    print("README results section updated")


if __name__ == "__main__":
    main()
