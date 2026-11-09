"""Build and execute the eight TalentScope notebooks.

    python scripts/build_notebooks.py            # build + execute all (notebook 05 re-runs the full experiment, ~10 min)
    python scripts/build_notebooks.py 03 07      # only these

Every notebook imports the same ``talentscope`` package as the Streamlit app,
so the notebooks document the code that actually runs, rather than a copy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = '''import sys, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from talentscope import config
from talentscope.reporting import figures as F
F.setup()
print("TalentScope root:", ROOT.name, "| seed", config.RANDOM_SEED)'''


def md(s):
    return nbformat.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbformat.v4.new_code_cell(s.strip("\n"))


# =============================================================================================== 01
NB01 = [
    md("""
# 01 · Data understanding

**TalentScope notebook series** · data → EDA → preprocessing → imbalance → models → evaluation → explainability → deployment

This notebook introduces the IBM HR Analytics Employee Attrition & Performance dataset, checks that the file on disk is
the one documented, and runs the same validation pipeline the dashboard's *Data laboratory* page uses
(`talentscope.data.validate_dataset`).

> **The data is synthetic.** IBM data scientists created it to resemble an HR system. It contains no real employees,
> and patterns found in it are not evidence about any real organisation.
"""),
    code(SETUP),
    md("""
## 1. Source and version

The CSV is committed at `data/raw/ibm_hr_employee_attrition.csv`. It was taken from IBM's public
`employee-attrition-aif360` repository, which hosts the same file as the Kaggle dataset
(`pavansubhasht/ibm-hr-analytics-attrition-dataset`). The SHA-256 hash below pins the exact version every result in
this project was computed on.
"""),
    code('''from talentscope.data import load_raw, dataset_card, schema
raw = load_raw()
card = dataset_card(raw)
pd.Series(card).to_frame("value")'''),
    md("""
## 2. Shape, columns and types
"""),
    code('''print("shape:", raw.shape)
raw.head()'''),
    code('''from talentscope.data.validation import column_profile
profile = column_profile(raw)
profile[["column", "role", "dtype", "unique", "missing", "min", "max", "description"]]'''),
    md("""
The 35 columns fall into six roles:

* **target**: `Attrition` (Yes / No)
* **identifier**: `EmployeeNumber`, a row ID with no predictive meaning
* **constants**: `EmployeeCount`, `Over18` and `StandardHours`, each holding a single value
* **14 numeric** attributes (age, pay rates, tenure and experience counts)
* **9 ordinal** survey and level codes (1 to 4 or 1 to 5, where order matters)
* **7 nominal** categories (department, role, travel, gender, marital status, education field, overtime)
"""),
    code('''profile["role"].value_counts().rename("columns").to_frame()'''),
    md("""
## 3. Target distribution
"""),
    code('''vc = raw["Attrition"].value_counts()
rate = vc["Yes"] / len(raw)
fig, ax = plt.subplots(figsize=(6, 1.6))
ax.barh([""], [vc["No"]], color=F.TEAL, label=f"Stayed {vc['No']:,} ({1 - rate:.1%})")
ax.barh([""], [vc["Yes"]], left=[vc["No"]], color=F.CLAY, label=f"Left {vc['Yes']:,} ({rate:.1%})")
ax.set_xlim(0, len(raw)); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2); ax.grid(False)
ax.set_title("Attrition in the full dataset"); plt.show()
print(f"Imbalance ratio {vc['No'] / vc['Yes']:.2f} : 1. A model that always predicts 'stay' is {1 - rate:.1%} accurate.")'''),
    md("""
About one employee in six left. That imbalance drives three decisions later in the series:

* Accuracy is not a useful headline metric.
* Splits must be stratified so every fold keeps the 16% rate.
* Rebalancing strategies are worth testing, though not assuming.
"""),
    md("""
## 4. Data quality: the validation report
"""),
    code('''from talentscope.data import validate_dataset
rep = validate_dataset(raw)
print(f"{rep.n_pass} passed, {rep.n_warn} warnings, {rep.n_fail} failures")
rep.to_frame()'''),
    md("""
**Missing values and duplicates.** There are no missing cells and no duplicate rows, including when `EmployeeNumber`
is ignored. The pipeline still contains imputers (fitted per training fold) so that it is safe on new records with gaps.
"""),
    code('''rep.missing.query("missing > 0") if rep.missing["missing"].sum() else "no missing values in any column"'''),
    md("""
**Constant and redundant columns.** Three columns carry one value for every employee. Several numeric pairs are strongly
correlated (job level and monthly income at r = 0.95). Correlated features are kept: none duplicates another, and the
models tolerate collinearity. The cost is that importance gets *shared* between them, which matters when reading SHAP
values in notebook 07.
"""),
    code('''print("constant columns:", rep.constant_columns)
rep.redundant_pairs'''),
    md("""
**Unexpected values.** Every categorical value appears in the data dictionary, and every numeric column falls inside a
plausible range. The logical constraints hold for every employee: role tenure ≤ company tenure ≤ total experience, and
manager tenure ≤ company tenure. One weakness is that `PerformanceRating` only takes the values 3 and 4, so it can
carry little signal.
"""),
    code('''display(rep.unexpected_values if len(rep.unexpected_values) else "no undocumented categories")
display(rep.out_of_range if len(rep.out_of_range) else "no out-of-range values")
raw["PerformanceRating"].value_counts().sort_index().rename("employees").to_frame()'''),
    md("""
## 5. Leakage screen

A feature that almost perfectly separates leavers from stayers on its own would point to label leakage, such as a
post-exit field. Each raw feature is scored alone with univariate ROC-AUC. Categorical features are encoded by their
observed attrition rate for this screen only; that encoding is never used for modelling.
"""),
    code('''ls = rep.leakage_screen.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(ls["feature"], ls["univariate_auc"], color=F.BLUE)
ax.axvline(0.5, color=F.MUTED, lw=1); ax.axvline(0.9, color=F.CLAY, lw=1, ls=":")
ax.set_xlim(0.45, 1); ax.set_xlabel("univariate ROC-AUC (0.5 = no signal, 0.9 = suspicious)")
ax.set_title("Strongest single features"); ax.grid(axis="y", visible=False); plt.show()
print("max univariate AUC:", rep.leakage_screen["univariate_auc"].max())'''),
    md("""
The strongest single feature reaches an AUC of about 0.68, far from the 0.9 alarm line, so no column behaves like a
proxy for the label. Prediction will have to come from combining many moderate signals.

## 6. What is kept for modelling
"""),
    code('''from talentscope.data import prepare_dataset
prepared = prepare_dataset(raw)
print("dropped:", schema.DROP_COLUMNS)
print("model features:", len(schema.MODEL_FEATURES))
pd.DataFrame({"numeric": pd.Series(schema.NUMERIC_FEATURES), "ordinal": pd.Series(schema.ORDINAL_FEATURES),
              "nominal": pd.Series(schema.NOMINAL_FEATURES)}).fillna("")'''),
    md("""
## Summary

| Check | Result |
|---|---|
| Rows × columns | 1,470 × 35, as documented |
| Target | 237 left (16.1%), 1,233 stayed |
| Missing / duplicates | none / none |
| Dropped | `EmployeeNumber` (identifier), `EmployeeCount`, `Over18`, `StandardHours` (constants) |
| Leakage | no single feature above 0.68 AUC |
| Watch-outs | small positive class; correlated pay, level and tenure features; `PerformanceRating` has 2 levels |

**Limitations.** 1,470 synthetic rows is a small sample. A held-out set of 20% contains only about 47 leavers, so every
held-out metric will carry a wide confidence interval. Notebook 06 quantifies how wide.
"""),
]

# =============================================================================================== 02
NB02 = [
    md("""
# 02 · Exploratory data analysis

Univariate, bivariate and multivariate views of the workforce and of attrition. Every rate is reported with its
group size and a 95% Wilson interval, and every test result with a Benjamini-Hochberg adjusted p-value. The functions
are those behind the dashboard's *Workforce* and *Attrition* pages (`talentscope.analytics`).

> These are **associations in synthetic data**. None of them shows that changing an attribute would change whether
> someone leaves.
"""),
    code(SETUP),
    code('''from talentscope.data import load_dataset, schema
from talentscope.analytics import (workforce_kpis, rate_by, association_ranking, numeric_comparison,
                                   headline_findings, correlation_matrix, cramers_v_matrix, crosstab_rates, DIMENSIONS)
df = load_dataset()
k = workforce_kpis(df)
pd.Series({key: v for key, v in k.items() if key != "rate_ci"}).to_frame("value")'''),
    md("## 1. Univariate: the workforce"),
    code('''num = ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome", "NumCompaniesWorked",
       "YearsSinceLastPromotion", "PercentSalaryHike", "TrainingTimesLastYear"]
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for ax, c in zip(axes.flat, num):
    ax.hist(df[c], bins=25, color=F.BLUE, alpha=0.85)
    ax.set_title(f"{schema.PRETTY_NAMES[c]}  (skew {df[c].skew():.2f})")
fig.suptitle("Numeric attributes", x=0.01, ha="left", fontweight="bold"); fig.tight_layout(rect=(0, 0, 1, 0.97)); plt.show()'''),
    md("""
Income, tenure, total experience and years since promotion are right-skewed: most employees are early in their
careers, with a long tail of senior, long-serving staff. Age is roughly symmetric around 37.
"""),
    code('''fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, c in zip(axes.flat, ["Department", "JobRole", "BusinessTravel", "MaritalStatus", "EducationField", "OverTime"]):
    vc = df[c].value_counts()
    ax.barh(vc.index[::-1], vc.values[::-1], color=F.BLUE); ax.set_title(schema.PRETTY_NAMES[c])
    ax.grid(axis="y", visible=False)
fig.tight_layout(rect=(0, 0, 1, 0.97)); plt.show()'''),
    md("## 2. Bivariate: attrition rate by group"),
    code('''overall = df["AttritionFlag"].mean()
dims = ["department", "job_role", "overtime", "age", "income", "tenure", "job_satisfaction", "work_life", "travel",
        "promotion", "marital", "stock"]
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for ax, key in zip(axes.flat, dims):
    col, label, ordered = DIMENSIONS[key]
    F.rate_dots(ax, rate_by(df, col), overall, f"by {label.lower()}", ordered=ordered)
fig.suptitle(f"Attrition rate with 95% Wilson intervals · vertical line = overall {overall:.1%}", x=0.01, ha="left",
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97)); plt.show()'''),
    code('''for f in headline_findings(df, k=6):
    print("•", f["text"], f"({f['ratio']:.1f}x)")'''),
    md("""
**Observations**

* **Overtime** is the clearest single gap: about 31% of overtime workers left against 10% of everyone else.
* **Career stage** dominates the rest. Attrition is highest under 26, in the first year or two at the company, at job
  level 1, in the lowest income band and among employees with no stock options. These attributes describe overlapping
  groups of junior staff, so they are not independent findings.
* **Sales Representatives** (40%) and **Laboratory Technicians** (24%) have the highest role rates. Directors and
  managers have the lowest, at around 3% to 7%.
* **Single** employees and **frequent travellers** leave more often.
* **Null or weak results:** promotion recency shows no consistent pattern, since the intervals overlap. Satisfaction
  scores matter mainly at the "Low" level, and `PerformanceRating` (two levels) shows almost no difference.
"""),
    code('''pd.concat({"Performance rating": rate_by(df, "PerformanceRating"), "Gender": rate_by(df, "Gender")})[
    ["group", "n", "left", "rate", "ci_low", "ci_high"]]'''),
    md("## 3. Statistical tests across all 30 attributes"),
    code('''assoc = association_ranking(df)
assoc[["label", "test", "effect_name", "effect", "direction", "p_value", "p_adjusted", "significant"]]'''),
    code('''a = assoc.iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(a["label"], a["effect"], color=[F.CLAY if s else F.NEUTRAL for s in a["significant"]])
ax.set_xlabel("effect size (Cramer's V or |rank-biserial r|)")
ax.set_title("Association with attrition · grey = not significant after BH correction"); ax.grid(axis="y", visible=False)
plt.show()
print(f"{assoc['significant'].sum()} of {len(assoc)} attributes significant at BH-adjusted p < 0.05")'''),
    md("""
Most attributes are significantly associated with attrition, but every effect is **small to moderate**: the largest is
around 0.3. That matches the leakage screen in notebook 01. No single attribute is decisive, so a model has to combine
them.

Ten attributes are *not* significant after correction: performance rating, hourly and monthly rate, gender, salary
hike, education, number of companies worked, relationship satisfaction, trainings last year and years since
promotion. They all stay in the model, because dropping features based on full-data tests would leak information from
the test set into feature selection.

Two of them, **years since promotion** and **companies worked**, matter a lot to the model in notebook 07 despite
weak univariate effects. Once career stage is held fixed, a long wait for promotion or frequent job changes become
informative. A univariate test cannot see that kind of conditional effect.
"""),
    code('''numeric_comparison(df).sort_values("rank_biserial")[["label", "median_left", "median_stayed", "rank_biserial", "p_adjusted"]]'''),
    md("## 4. Multivariate structure"),
    code('''cm = correlation_matrix(df)
labels = [schema.PRETTY_NAMES.get(c, "Attrition") for c in cm.columns]
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=90); ax.set_yticks(range(len(labels)), labels); ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho"); ax.set_title("Rank correlations"); plt.show()'''),
    md("""
A tight block links **age, total working years, job level, monthly income and the four tenure counts**. That block is
the "career stage" factor behind most of the univariate gaps above. Satisfaction surveys are nearly uncorrelated with
each other and with the career block, so they add independent information.
"""),
    code('''cv = cramers_v_matrix(df)
cv.index = cv.columns = [schema.PRETTY_NAMES.get(c, c) for c in cv.columns]
cv.round(2)'''),
    code('''rates, counts = crosstab_rates(df, "JobRole", "OverTime", min_n=10)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(rates.values, cmap="Reds", vmin=0, vmax=0.7)
for i in range(rates.shape[0]):
    for j in range(rates.shape[1]):
        v = rates.values[i, j]
        ax.text(j, i, "" if np.isnan(v) else f"{v:.0%}\\nn={counts.values[i, j]}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.35 else F.INK)
ax.set_xticks([0, 1], [f"Overtime = {c}" for c in rates.columns]); ax.set_yticks(range(len(rates)), rates.index); ax.grid(False)
ax.set_title("Attrition by job role and overtime"); plt.show()'''),
    md("""
**Interaction.** Overtime is associated with higher attrition in *every* role, so the overtime gap is not just a role
effect. Sales Representatives who work overtime have the highest rate in the table. Their cell is small, though, so the
exact figure is uncertain.

## Summary

* Attrition concentrates among junior, early-tenure, lower-paid employees, among those working overtime, and in a few
  roles (Sales Representative, Laboratory Technician).
* All effects are moderate, and many attributes overlap, so a multivariate model is needed and should be expected to be
  imperfect.
* Weak or null attributes: promotion recency, performance rating, gender, education and the three "rate" pay fields.
"""),
]

# =============================================================================================== 03
NB03 = [
    md("""
# 03 · Data preprocessing and leakage prevention

How raw rows become model inputs, and the steps that keep the held-out test set and the validation folds untouched.
Everything here is implemented in `talentscope.preprocessing` and `talentscope.modeling.registry`, the same code the
dashboard trains with.
"""),
    code(SETUP),
    md("""
## 1. One split, made first

The stratified 80/20 split happens **before** any imputer, scaler, encoder or sampler is fitted. The seed is fixed, so
every notebook and the app see the same 294 test employees.
"""),
    code('''from talentscope.data import load_dataset, split_features_target, schema
from talentscope.preprocessing import make_split
df = load_dataset()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = make_split(X, y)
pd.DataFrame({"rows": [len(y_train), len(y_test)], "left": [y_train.sum(), y_test.sum()],
              "attrition rate": [y_train.mean(), y_test.mean()]}, index=["train", "test"])'''),
    code('''assert set(X_train.index).isdisjoint(X_test.index)
print("test rows overlap with training rows:", len(set(X_train.index) & set(X_test.index)))'''),
    md("""
## 2. The pipeline

```
impute  ->  [resample]  ->  encode  ->  model
```

* **impute**: median for numeric and ordinal columns, most frequent value for nominal ones.
* **resample** (optional): SMOTENC or random oversampling. `imblearn` runs samplers only while fitting.
* **encode**: round integer-coded columns (needed after SMOTENC), standardise them for scale-sensitive models
  (logistic regression, SVM), and one-hot encode nominal columns, ignoring unseen categories.
"""),
    code('''from talentscope.modeling import build_pipeline
pipe = build_pipeline("logreg", "smotenc", seed=42, y=y_train)
pipe'''),
    code('''pipe.fit(X_train, y_train)
from talentscope.preprocessing import feature_names, transform_features
names = feature_names(pipe)
Z = transform_features(pipe, X_test)
print("raw features:", X_train.shape[1], "-> model inputs:", len(names))
pd.Series(names[:8] + ["..."] + names[-6:], name="transformed feature names")'''),
    md("""
## 3. Fitted on training rows only

Without resampling, the scaler's learned means equal the **training** means, not the means over all 1,470 employees.
With SMOTENC in the pipeline, the scaler is fitted *after* resampling, on real plus synthetic training rows. Its means
then shift towards the leavers (younger, lower paid), but still contain nothing from the test set.
"""),
    code('''cols = schema.NUMERIC_FEATURES + schema.ORDINAL_FEATURES
plain = build_pipeline("logreg", "none", seed=42, y=y_train).fit(X_train, y_train)
scale_plain = plain.named_steps["encode"].named_transformers_["num"].named_steps["scale"]
scale_smote = pipe.named_steps["encode"].named_transformers_["num"].named_steps["scale"]
check = pd.DataFrame({"scaler mean (no resampling)": scale_plain.mean_, "train mean": X_train[cols].mean().values,
                      "all-rows mean": X[cols].mean().values,
                      "scaler mean (after SMOTENC)": scale_smote.mean_}, index=cols).head(6)
assert np.allclose(scale_plain.mean_, X_train[cols].mean().values)
check'''),
    md("## 4. Missing values and unseen categories"),
    code('''probe = X_test.head(3).copy()
probe.iloc[0, probe.columns.get_loc("MonthlyIncome")] = np.nan
probe.iloc[1, probe.columns.get_loc("JobRole")] = np.nan
probe.iloc[2, probe.columns.get_loc("Department")] = "Brand New Department"
imputed = pipe.named_steps["impute"].transform(probe)
print("imputed income:", imputed.iloc[0]["MonthlyIncome"], "| training median:", X_train["MonthlyIncome"].median())
print("imputed role  :", imputed.iloc[1]["JobRole"], "| training mode:", X_train["JobRole"].mode()[0])
print("probabilities :", pipe.predict_proba(probe)[:, 1].round(3))'''),
    md("""
## 5. SMOTENC output is valid data

SMOTENC interpolates numeric columns. That would create a synthetic employee with, say, a job satisfaction of 2.37.
The `round` step restores whole-number codes, and nominal columns only ever take values that already exist.
"""),
    code('''Xi = pipe.named_steps["impute"].transform(X_train)
Xr, yr = pipe.named_steps["sampler"].fit_resample(Xi, y_train)
synthetic = Xr.iloc[len(X_train):]
print("rows before / after:", len(Xi), "/", len(Xr), "| synthetic leavers:", len(synthetic))
before_round = synthetic[["JobSatisfaction", "JobLevel", "Age"]].head(4)
after_round = pd.DataFrame(np.rint(before_round.values), columns=before_round.columns, index=before_round.index)
pd.concat({"interpolated": before_round, "after rounding": after_round}, axis=1)'''),
    md("""
## 6. Why the order matters: a leakage experiment

Two ways to estimate cross-validated PR-AUC for the same logistic regression:

* **Wrong:** oversample the whole training set with SMOTENC first, then cross-validate. Synthetic copies built from
  validation employees leak into the training folds.
* **Right:** put SMOTENC inside the pipeline, so it is refitted on the training part of each fold.
"""),
    code('''from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
cv = StratifiedKFold(5, shuffle=True, random_state=0)

right = cross_val_score(build_pipeline("logreg", "smotenc", 42, y_train), X_train, y_train, cv=cv, scoring="average_precision")

Xi = build_pipeline("logreg", "none", 42, y_train).named_steps["impute"].fit_transform(X_train)
X_leak, y_leak = SMOTENC(categorical_features=schema.NOMINAL_FEATURES, random_state=42).fit_resample(Xi, y_train)
no_sampler = build_pipeline("logreg", "none", 42, y_train)
wrong = cross_val_score(no_sampler, X_leak, y_leak, cv=cv, scoring="average_precision")

res = pd.DataFrame({"mean CV PR-AUC": [right.mean(), wrong.mean()], "SD": [right.std(), wrong.std()]},
                   index=["SMOTENC inside each fold (correct)", "SMOTENC before CV (leaky)"])
res'''),
    code('''gap = wrong.mean() - right.mean()
print(f"Resampling before cross-validation inflates PR-AUC by {gap:+.3f}. Part of that comes from evaluating on a")
print("50/50 resampled validation fold (PR-AUC depends on prevalence), and part from synthetic neighbours of")
print("validation employees sitting in the training folds. Either way, the leaky number does not describe")
print("performance on real employees at the real 16% attrition rate.")'''),
    md("""
## Summary

* The split comes first and is stratified; test rows never reach a `fit` call.
* Imputation, rounding, scaling, one-hot encoding and resampling are all learned inside each training fold.
* New categories and missing values are handled at prediction time.
* Resampling before cross-validation produces a much higher, and meaningless, score. The pipeline makes that mistake
  impossible.
"""),
]

# =============================================================================================== 04
NB04 = [
    md("""
# 04 · Class imbalance and SMOTE / SMOTENC

The training set has about 5.2 stayers per leaver. This notebook shows what the imbalance does to naive models, how
SMOTENC works on mixed numeric and categorical data, and how SMOTENC compares with class weighting, random
oversampling and doing nothing. The comparison is read from the committed default experiment, which notebook 05
re-runs from scratch.

**Spoiler, stated up front:** in this dataset SMOTENC did *not* improve ranking quality (PR-AUC). It lowered PR-AUC for
five of six algorithms, and the sixth (a single decision tree) gained 0.004, well inside fold noise. That negative
result is reported as is.
"""),
    code(SETUP),
    code('''from talentscope.data import load_dataset, split_features_target, schema
from talentscope.preprocessing import make_split
from talentscope.modeling import build_pipeline, load_bundle
from talentscope.evaluation import compute_metrics
df = load_dataset(); X, y = split_features_target(df)
X_train, X_test, y_train, y_test = make_split(X, y)
b = load_bundle()
print("training set:", int((y_train == 0).sum()), "stayed /", int(y_train.sum()), "left")'''),
    md("## 1. What imbalance does to a naive model"),
    code('''from sklearn.model_selection import cross_val_predict, StratifiedKFold
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = {}
for name, model, strat in [("Dummy (prior)", "dummy", "none"), ("Logistic, no rebalancing", "logreg", "none"),
                           ("Logistic, class weights", "logreg", "class_weight"), ("Logistic, SMOTENC", "logreg", "smotenc")]:
    p = cross_val_predict(build_pipeline(model, strat, 42, y_train), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    m = compute_metrics(y_train, p, 0.5)
    rows[name] = {k: m[k] for k in ("accuracy", "recall", "precision", "pr_auc", "brier")} | {"mean predicted": p.mean()}
pd.DataFrame(rows).T'''),
    md("""
* The dummy model scores about 84% **accuracy** while catching **no** leavers, which is why accuracy is not used to
  choose models.
* At the default 0.5 threshold, the unweighted logistic regression is conservative: it finds a minority of leavers.
* Class weights and SMOTENC push predicted probabilities upwards. Recall at 0.5 jumps, precision falls, and the mean
  predicted probability rises well above the true 16%, so calibration gets worse (Brier rises).
* PR-AUC, which does not depend on the threshold, barely moves or falls. **Rebalancing mostly moves the threshold; it
  does not make the model rank employees better.**
"""),
    md("""
## 2. How SMOTENC works

For each leaver in the training fold, SMOTENC picks one of its *k* nearest leaver neighbours. It creates a synthetic
leaver whose numeric features lie on the line between the two, and whose nominal features take the most common value
among the neighbours. Numeric distances use standardised values; nominal mismatches add a penalty based on the median
numeric standard deviation.

The plot projects one training fold onto its first two principal components (fitted on real rows only).
"""),
    code('''from sklearn.decomposition import PCA
pipe = build_pipeline("logreg", "smotenc", 42, y_train)
fold_tr, fold_va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
Xf, yf = X_train.iloc[fold_tr], y_train.iloc[fold_tr]
Xi = pipe.named_steps["impute"].fit_transform(Xf)
Xr, yr = pipe.named_steps["sampler"].fit_resample(Xi, yf)
enc = pipe.named_steps["encode"].fit(Xi)
Zr = enc.transform(Xr)
pca = PCA(2, random_state=0).fit(enc.transform(Xi))
P = pca.transform(Zr); n = len(Xi); yr = np.asarray(yr)
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(*P[:n][yr[:n] == 0].T, s=8, color=F.NEUTRAL, alpha=0.35, label="stayed (real)")
ax.scatter(*P[n:].T, s=10, color=F.AMBER, alpha=0.55, label="left (synthetic)")
ax.scatter(*P[:n][yr[:n] == 1].T, s=16, color=F.CLAY, alpha=0.9, label="left (real)")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(); ax.set_title("One training fold after SMOTENC"); plt.show()
print("fold before:", np.bincount(yf), "| after:", np.bincount(yr), "| validation fold (untouched):", np.bincount(y_train.iloc[fold_va]))'''),
    md("""
Synthetic leavers fill the space *between* real leavers. In this dataset leavers overlap heavily with stayers, so the
synthetic points also land inside the stayers' region. The boundary moves towards the stayers (more leavers flagged),
but the ordering of real employees improves little.

## 3. The controlled comparison (default experiment)

Each of six algorithms was tuned and evaluated under four strategies with identical folds, seeds and search budgets.
The numbers are mean PR-AUC over 15 cross-validation folds on the training set.
"""),
    code('''cv_sum = b.cv_summary
piv = cv_sum.pivot_table(index="model_name", columns="strategy_name", values="cv_pr_auc_mean")
piv = piv[["No rebalancing", "Class weights", "SMOTENC", "Random oversampling"]].sort_values("No rebalancing", ascending=False)
piv.style.format("{:.3f}").background_gradient(cmap="Greens", axis=None)'''),
    code('''base = cv_sum[cv_sum.strategy == "none"].set_index("model")
rows = []
for _, r in cv_sum[cv_sum.strategy != "none"].iterrows():
    b0 = base.loc[r["model"]]
    se = np.sqrt((r["cv_pr_auc_std"] ** 2 + b0["cv_pr_auc_std"] ** 2) / r["n_folds"])
    rows.append({"algorithm": r["model_name"], "strategy": r["strategy_name"],
                 "Δ PR-AUC": r["cv_pr_auc_mean"] - b0["cv_pr_auc_mean"], "≈2 SE": 2 * se,
                 "Δ recall@0.5": r["cv_recall_mean"] - b0["cv_recall_mean"],
                 "Δ Brier": r["cv_brier_mean"] - b0["cv_brier_mean"]})
delta = pd.DataFrame(rows)
delta.sort_values(["strategy", "Δ PR-AUC"])'''),
    code('''fig, ax = plt.subplots(figsize=(8, 4.2))
strategies = ["Class weights", "SMOTENC", "Random oversampling"]
colors = [F.BLUE, F.TEAL, F.AMBER]
algos = delta["algorithm"].unique()
for i, (s, c) in enumerate(zip(strategies, colors)):
    d = delta[delta.strategy == s].set_index("algorithm").reindex(algos)
    ax.scatter(d["Δ PR-AUC"], np.arange(len(algos)) + (i - 1) * 0.22, color=c, s=40, label=s, zorder=3)
ax.axvline(0, color=F.MUTED, lw=1)
ax.set_yticks(range(len(algos)), algos); ax.set_xlabel("change in mean CV PR-AUC vs no rebalancing")
ax.legend(loc="lower left"); ax.set_title("Did rebalancing improve ranking quality?"); ax.grid(axis="y", visible=False); plt.show()
sm = delta[delta.strategy == "SMOTENC"]
print(f"SMOTENC improved PR-AUC for {(sm['Δ PR-AUC'] > 0).sum()} of {len(sm)} algorithms; "
      f"median change {sm['Δ PR-AUC'].median():+.3f}. It raised recall at 0.5 for {(sm['Δ recall@0.5'] > 0).sum()} of {len(sm)}.")'''),
    md("""
**Findings (from the saved artefacts above)**

1. **SMOTENC lowered mean CV PR-AUC for five of six algorithms**, four of them by more than two standard errors. The
   exception, the single decision tree, moved by +0.004, which is noise.
2. **Class weights** and **random oversampling** were close to neutral for most algorithms. They were sometimes
   marginally lower and rarely higher, and the differences sat inside fold-to-fold noise.
3. All three strategies raise **recall at the 0.5 threshold** and worsen the **Brier score**. They shift predicted
   probabilities up, much as lowering the threshold would, without that step's transparency.
4. The best configuration overall used **no rebalancing**, with a threshold tuned on out-of-fold predictions (notebook
   06). That gives the recall benefit while keeping probabilities calibrated.

Why might SMOTENC hurt here? With about 150 leavers per training fold and 51 one-hot inputs, the nearest leaver
neighbours are often far apart. Interpolating between them creates implausible employees in regions dominated by
stayers, which adds label noise near the boundary.

## 4. Sensitivity of SMOTENC to its settings

Is the negative result just a bad configuration? The check below runs a small grid over the sampling ratio and *k* for
logistic regression, with fixed C, using the same CV folds.
"""),
    code('''from sklearn.model_selection import cross_val_score
cv5 = StratifiedKFold(5, shuffle=True, random_state=7)
C = b.best_params["logreg|none"]["C"]
ref = cross_val_score(build_pipeline("logreg", "none", 42, y_train).set_params(model__C=C), X_train, y_train,
                      cv=cv5, scoring="average_precision").mean()
grid = []
for ratio in (0.35, 0.5, 0.75, 1.0):
    for k in (3, 5, 10):
        p = build_pipeline("logreg", "smotenc", 42, y_train, smote_sampling_strategy=ratio, smote_k_neighbors=k).set_params(model__C=C)
        grid.append({"ratio": ratio, "k": k, "PR-AUC": cross_val_score(p, X_train, y_train, cv=cv5, scoring="average_precision").mean()})
g = pd.DataFrame(grid).pivot(index="ratio", columns="k", values="PR-AUC")
print(f"no rebalancing (same C, same folds): {ref:.3f}")
g.style.format("{:.3f}")'''),
    code('''print(f"best SMOTENC setting: {g.max().max():.3f} vs {ref:.3f} without rebalancing "
      f"({g.max().max() - ref:+.3f}). Milder ratios hurt less, but none beat the baseline here.")'''),
    md("""
The pattern holds across settings. Gentler oversampling (a lower ratio) does less damage, but no configuration in the
grid clearly beats no rebalancing. SMOTENC stays available in the Model laboratory so the comparison can be repeated.

## Summary

| Strategy | Effect on PR-AUC (ranking) | Effect at 0.5 threshold | Calibration |
|---|---|---|---|
| None + tuned threshold | best overall | threshold chosen on OOF data | best |
| Class weights | about neutral | recall up, precision down | worse |
| Random oversampling | neutral to slightly lower | recall up | worse |
| SMOTENC | lower for 5 of 6 algorithms; no meaningful gain anywhere | recall up | worse |

Resampling only ever touched training folds; the evidence is in `experiments/default/resampling.json`.
"""),
]

# =============================================================================================== 05
NB05 = [
    md("""
# 05 · Model training and comparison

This notebook **re-runs the complete default experiment from scratch** and checks the result against the artefacts
committed in `experiments/default`, which the Streamlit app reads.

| Item | Setting |
|---|---|
| Algorithms | Dummy, logistic regression, decision tree, random forest, SVM, XGBoost, LightGBM |
| Strategies | none, class weights, SMOTENC, random oversampling |
| Configurations | 25 (the dummy runs once) |
| Search | `RandomizedSearchCV`, 30 candidates, 5-fold stratified CV on the training split, scoring PR-AUC |
| Comparison | tuned configurations re-scored with 3 × 5-fold repeated stratified CV (a different seed) |
| Selection | one-standard-error rule on mean CV PR-AUC |
| Threshold | maximise F1 on out-of-fold training predictions |

Runtime is about 10 minutes on a laptop.
"""),
    code(SETUP),
    code('''import time
from talentscope.modeling import ExperimentConfig, run_experiment, load_bundle, MODEL_SPECS
cfg = ExperimentConfig(name="default")
t0 = time.time()
result = run_experiment(cfg)
print(f"finished in {time.time() - t0:.0f} s")'''),
    md("## 1. Reproducibility check against the committed artefacts"),
    code('''saved = load_bundle()
a = result.cv_summary.set_index("combo").sort_index()
s = saved.cv_summary.set_index("combo").sort_index()
cols = ["cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "cv_brier_mean", "oof_threshold"]
diff = (a[cols] - s[cols]).abs().max()
print("largest absolute difference per column:\\n", diff.round(8))
print("selected now:", result.selection["combo"], "| committed:", saved.selection["combo"])
print("held-out PR-AUC now / committed:", round(result.test_metrics["pr_auc"], 6), "/", round(saved.test_metrics["pr_auc"], 6))
assert diff.max() < 1e-6 and result.selection["combo"] == saved.selection["combo"]'''),
    md("""
The fresh run matches the committed artefacts to rounding precision: same scores, same selected model, same threshold
and same test result. Every number in the app can therefore be regenerated with `python scripts/train_default.py`.

## 2. Cross-validated comparison
"""),
    code('''table = result.cv_summary[["model_name", "strategy_name", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
                           "cv_brier_mean", "oof_f1", "oof_threshold", "fit_seconds", "selected"]]
table.style.format({c: "{:.3f}" for c in ["cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "cv_brier_mean", "oof_f1"]})'''),
    code('''r = result.cv_summary.sort_values("cv_pr_auc_mean")
fig, ax = plt.subplots(figsize=(8, 8))
yy = np.arange(len(r))
ax.errorbar(r["cv_pr_auc_mean"], yy, xerr=r["cv_pr_auc_std"], fmt="none", ecolor=F.GRID, elinewidth=2)
ax.scatter(r["cv_pr_auc_mean"], yy, c=[F.STRATEGY_COLORS[s] for s in r["strategy"]], s=[90 if s else 36 for s in r["selected"]],
           marker="o", zorder=3, edgecolor="white")
ax.set_yticks(yy, [("★ " if s else "") + f"{m} · {t}" for m, t, s in zip(r["model_name"], r["strategy_name"], r["selected"])])
for key, label in [("none", "No rebalancing"), ("class_weight", "Class weights"), ("smotenc", "SMOTENC"), ("random_oversample", "Random oversampling")]:
    ax.scatter([], [], color=F.STRATEGY_COLORS[key], label=label)
ax.legend(loc="lower right"); ax.set_xlabel("mean CV PR-AUC ± SD (15 folds)"); ax.grid(axis="y", visible=False)
ax.set_title("All 25 configurations"); plt.show()'''),
    md("""
**Reading the ranking**

* **Every model clearly beats the dummy baseline** (PR-AUC about 0.16, the prevalence).
* **The simplest model wins.** L2-regularised logistic regression without rebalancing has the highest mean PR-AUC. SVM and XGBoost follow within a fraction of a standard deviation.
* **Random forest and the single decision tree trail.** The tree in particular overfits small leaf sets; even tuned, it
  sits near 0.36.
* **Fold-to-fold SD (about 0.06) is large next to the gaps between the top configurations** (about 0.005 to 0.02).
  Many of these differences are not real.
"""),
    code('''folds = result.cv_folds
top = result.cv_summary.head(6)["combo"].tolist()
fig, ax = plt.subplots(figsize=(10, 4))
ax.boxplot([folds[folds.combo == c]["pr_auc"] for c in top], widths=0.5, medianprops=dict(color=F.CLAY))
for i, c in enumerate(top, 1):
    v = folds[folds.combo == c]["pr_auc"]
    ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(v)), v, s=12, color=F.BLUE, alpha=0.7, zorder=3)
ax.set_xticks(range(1, len(top) + 1), [c.replace("|", "\\n") for c in top], fontsize=8)
ax.set_ylabel("fold PR-AUC"); ax.set_title("Fold-level PR-AUC, top six configurations"); plt.show()'''),
    md("## 3. Is the winner really better? Paired fold differences"),
    code('''from scipy import stats
sel = result.selection["combo"]
piv = folds.pivot(index="fold", columns="combo", values="pr_auc")
rows = []
for c in top[1:]:
    d = piv[sel] - piv[c]
    t = stats.ttest_1samp(d, 0)
    rows.append({"versus": c, "mean diff": d.mean(), "SD diff": d.std(ddof=1), "folds selected wins": int((d > 0).sum()),
                 "paired t p-value": t.pvalue})
pd.DataFrame(rows)'''),
    md("""
Folds share training rows, so a paired t-test on CV folds is optimistic. Even so, the selected model's advantage over
the runners-up is small and inconsistent from fold to fold. This is the case the **one-standard-error rule** is for:
among configurations within one standard error of the best, take the simplest. Here the best and the simplest coincide,
so the rule confirms logistic regression rather than overriding it.

## 4. The search itself
"""),
    code('''tr = result.search_trials[result.search_trials.combo == "logreg|none"].copy()
tr["C"] = tr["params"].map(lambda s: __import__("json").loads(s)["C"])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.errorbar(tr["C"], tr["mean_cv_pr_auc"], yerr=tr["std_cv_pr_auc"], fmt="o", color=F.TEAL, ecolor=F.GRID, ms=5)
ax.set_xscale("log"); ax.set_xlabel("C (inverse regularisation strength)"); ax.set_ylabel("search CV PR-AUC")
ax.axvline(result.selection["params"]["C"], color=F.CLAY, lw=1)
ax.set_title("Logistic regression: 30 sampled values of C"); plt.show()
pd.Series(result.best_params).map(lambda p: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}).to_frame("tuned parameters")'''),
    md("""
Search PR-AUC is nearly flat from about C = 0.05 to C = 8 (0.644 to 0.654) and falls only under strong regularisation.
The model is not sensitive to the exact regularisation strength, which is reassuring for its stability.

## 5. Unsuccessful experiments, kept on the record

* **SMOTENC** reduced PR-AUC for five of six algorithms and gave no meaningful gain for the sixth (notebook 04).
* **Random forest** underperformed gradient boosting and logistic regression despite tuning depth, leaf size and
  feature sampling.
* **A single decision tree** is far behind (about 0.36). It is kept as an interpretable reference, not a candidate.
* **More complex models did not help.** XGBoost, LightGBM and the SVM all landed within noise of logistic regression or
  below it.

## Summary

The selected configuration is **logistic regression with no rebalancing**, with *C* tuned by randomised search and a
threshold to be chosen on out-of-fold predictions. Its held-out evaluation is the subject of notebook 06.
"""),
    code('''pd.Series({k: v for k, v in result.selection.items() if k not in ("reason",)}).to_frame("selection")'''),
]

# =============================================================================================== 06
NB06 = [
    md("""
# 06 · Model evaluation

The selected pipeline (logistic regression, no rebalancing), evaluated with the full protocol:

* **Threshold:** chosen on out-of-fold training predictions.
* **Held-out estimate:** one evaluation on the untouched 294-employee test set.
* **Uncertainty:** stratified bootstrap confidence intervals around every held-out metric.
* **Further checks:** calibration, threshold trade-offs, a learning curve and error rates by group.
"""),
    code(SETUP),
    code('''from talentscope.modeling import load_bundle
from talentscope.evaluation import compute_metrics, threshold_sweep, select_threshold, bootstrap_ci, curve_points, calibration_table
from talentscope.data import load_dataset, split_features_target
from talentscope.preprocessing import make_split
b = load_bundle()
df = load_dataset(); X, y = split_features_target(df)
X_train, X_test, y_train, y_test = make_split(X, y)
model = b.load_model()
p_test = model.predict_proba(X_test)[:, 1]
assert np.allclose(p_test, b.test_predictions.set_index("row_id").loc[X_test.index, "probability"], atol=1e-6)
oof = b.oof[b.selection["combo"]].to_numpy(); y_oof = b.oof["actual_left"].to_numpy()
print(b.label, "| threshold", b.threshold)'''),
    md("## 1. Choosing the threshold on training data"),
    code('''sweep = threshold_sweep(y_oof, oof)
thr, best = select_threshold(y_oof, oof, "f1")
fig, ax = plt.subplots(figsize=(8, 4))
for col, c in (("precision", F.BLUE), ("recall", F.CLAY), ("f1", F.TEAL)):
    ax.plot(sweep["threshold"], sweep[col], color=c, lw=2, label=col)
ax.axvline(thr, color=F.INK, lw=1); ax.text(thr + 0.01, 0.95, f"F1-optimal {thr:.2f}", fontsize=9)
ax.axvline(0.5, color=F.MUTED, lw=1, ls=":"); ax.set_xlabel("threshold"); ax.legend(); ax.set_title("Out-of-fold trade-off (training data)")
plt.show()
print(f"selected threshold {thr:.2f} (OOF F1 {best:.3f}); saved threshold {b.threshold:.2f}")'''),
    md("""
The default 0.5 threshold is too high for a 16% positive class. The F1-optimal point is lower, trading some precision
for much better recall. Because it was chosen on out-of-fold predictions, the test set below is a fair check of it.

## 2. Held-out results, once
"""),
    code('''m = compute_metrics(y_test, p_test, b.threshold)
ci = bootstrap_ci(y_test, p_test, b.threshold, n_resamples=1000, seed=42)
ci.assign(estimate=ci["estimate"].round(3), ci_low=ci["ci_low"].round(3), ci_high=ci["ci_high"].round(3))[["label", "estimate", "ci_low", "ci_high"]]'''),
    code('''fig, ax = plt.subplots(figsize=(7, 3.8))
c2 = ci[ci.metric != "brier"].iloc[::-1]
ax.hlines(range(len(c2)), c2["ci_low"], c2["ci_high"], color=F.CLAY, lw=2.5, alpha=0.6)
ax.scatter(c2["estimate"], range(len(c2)), color=F.CLAY, zorder=3, s=40)
ax.set_yticks(range(len(c2)), c2["label"]); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.set_title(f"Held-out metrics with 95% bootstrap intervals (n = {len(y_test)}, {int(y_test.sum())} leavers)"); plt.show()
print(f"PR-AUC {m['pr_auc']:.3f} against a chance level of {y_test.mean():.3f}; ROC-AUC {m['roc_auc']:.3f}")'''),
    md("""
**Reading the uncertainty.** With only 47 leavers in the test set, the PR-AUC interval spans roughly ±0.12. The
held-out PR-AUC sits inside the range the cross-validation predicted (mean about 0.66, fold SD about 0.06), slightly
lower but consistent. Small held-out samples routinely produce gaps of this size, which is why model choices were made
on CV results, not on this number.
"""),
    code('''from talentscope.reporting.figures import TEAL, CLAY
cp, co = curve_points(y_test, p_test), curve_points(y_oof, oof)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].plot(co["fpr"], co["tpr"], color=TEAL, ls=":", label="cross-validated (training)")
axes[0].plot(cp["fpr"], cp["tpr"], color=CLAY, label="held-out test"); axes[0].plot([0, 1], [0, 1], color=F.GRID)
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].legend(); axes[0].set_title("ROC")
axes[1].step(co["recall"], co["precision"], color=TEAL, ls=":", where="post", label="cross-validated (training)")
axes[1].step(cp["recall"], cp["precision"], color=CLAY, where="post", label="held-out test")
axes[1].axhline(y_test.mean(), color=F.GRID); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].legend()
axes[1].set_title("Precision-recall"); plt.tight_layout(); plt.show()'''),
    code('''from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, (p_test >= b.threshold).astype(int))
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"]).plot(ax=ax, cmap="Greens", colorbar=False); ax.grid(False)
ax.set_title(f"Test set at threshold {b.threshold:.2f}"); plt.show()
print(f"Caught {cm[1,1]} of {cm[1].sum()} leavers; {cm[0,1]} of {cm[0].sum()} stayers flagged by mistake.")
b.classification_report'''),
    md("## 3. Calibration"),
    code('''fig, ax = plt.subplots(figsize=(5.5, 5))
for t, (yy, pp, bins) in {"cross-validated (10 bins)": (y_oof, oof, 10), "held-out (5 bins)": (y_test.to_numpy(), p_test, 5)}.items():
    c = calibration_table(yy, pp, bins)
    ax.plot(c["mean_predicted"], c["observed_rate"], "o-", label=t)
ax.plot([0, 1], [0, 1], color=F.GRID); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed attrition rate")
ax.legend(); ax.set_title("Reliability"); plt.show()
print(f"Brier score {m['brier']:.4f} vs {y_test.mean() * (1 - y_test.mean()):.4f} for always predicting the base rate")'''),
    md("""
Without rebalancing, the logistic regression's probabilities are roughly calibrated: a 30% score means about 30% of
such employees left. That is what lets the risk simulator report a probability and an empirical reliability range
instead of a bare label. A rebalanced model would overstate every probability.

## 4. Is more data likely to help? Learning curve (training split only)
"""),
    code('''from sklearn.model_selection import learning_curve, StratifiedKFold
from talentscope.modeling import build_pipeline
pipe = build_pipeline("logreg", "none", 42, y_train).set_params(model__C=b.best_params["logreg|none"]["C"])
sizes, tr_s, va_s = learning_curve(pipe, X_train, y_train, train_sizes=np.linspace(0.2, 1.0, 6),
                                   cv=StratifiedKFold(5, shuffle=True, random_state=3), scoring="average_precision", n_jobs=-1)
fig, ax = plt.subplots(figsize=(7, 4))
for s, c, lab in ((tr_s, F.BLUE, "training folds"), (va_s, F.TEAL, "validation folds")):
    ax.plot(sizes, s.mean(1), "o-", color=c, label=lab); ax.fill_between(sizes, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=c, alpha=0.15)
ax.set_xlabel("training rows"); ax.set_ylabel("PR-AUC"); ax.legend(); ax.set_title("Learning curve"); plt.show()
pd.DataFrame({"rows": sizes, "train PR-AUC": tr_s.mean(1), "validation PR-AUC": va_s.mean(1)})'''),
    md("""
Validation PR-AUC is still rising at the full training size, and the gap to training PR-AUC is narrowing slowly. More
employees would probably help, which reinforces the small-sample limitation. Using the full training split, with no
sub-sampling cap, was the right call.

## 5. Error rates by group (responsible-AI check)
"""),
    code('''gm = pd.read_csv(b.explain_dir / "group_metrics.csv")
gm[["attribute", "group", "n", "actual_leavers", "observed_rate", "flagged_rate", "recall_tpr", "false_positive_rate", "precision"]]'''),
    md("""
Group sizes in the test set are small, with some under 40 and few leavers, so differences of 10 to 20 points can arise
by chance. Two patterns are still worth flagging for anyone extending this work:

* **Young employees (18 to 25)** are flagged far more often, with a much higher false-positive rate. The model has
  learned "young ⇒ risk" from career-stage attributes.
* **Employees aged 46 and over** have low recall: the few who leave are mostly missed.

Any real deployment would need a fairness review on real outcomes. Removing protected attributes (age, gender, marital
status) does not by itself fix this, because tenure, level and income carry similar information.

## Summary

| | Value |
|---|---|
| Threshold | chosen on out-of-fold predictions |
| Held-out PR-AUC | see the table above, with its 95% interval |
| Calibration | close to the diagonal; Brier below the base-rate model |
| Main limitation | 47 test leavers, so wide intervals; the learning curve suggests more data would help |
"""),
]

# =============================================================================================== 07
NB07 = [
    md("""
# 07 · Explainable AI with SHAP

Global and local explanations of the selected pipeline, computed on the held-out test set:

* **SHAP values** from `shap.LinearExplainer`, the exact explainer for a linear model, with a training-set background.
* **Permutation importance**, an independent check: the drop in held-out PR-AUC when a feature is shuffled.

Both describe **what the model does**. Neither says what causes attrition.
"""),
    code(SETUP),
    code('''import shap
from talentscope.modeling import load_bundle
from talentscope.explainability import build_explainer, explain, aggregate_by_source, load_explanations
from talentscope.data import load_dataset, split_features_target, schema
from talentscope.preprocessing import make_split
b = load_bundle(); model = b.load_model()
df = load_dataset(); X, y = split_features_target(df)
X_train, X_test, y_train, y_test = make_split(X, y)
saved = load_explanations(b)
print("explainer:", saved["meta"]["explainer"], "| output space:", saved["meta"]["output_space"], "| rows:", saved["meta"]["rows_explained"])'''),
    md("## 1. Recompute SHAP and check it against the saved values and the model"),
    code('''explainer = build_explainer(model, b.selection["model"], X_train)
ex = explain(model, explainer, X_test, b.selection["model"])
print("matches saved values:", np.allclose(ex.values, saved["shap"].values, atol=1e-8))
logit = ex.base_value + ex.values.sum(1)
print("base + sum(SHAP) reproduces predictions:", np.allclose(1 / (1 + np.exp(-logit)), model.predict_proba(X_test)[:, 1], atol=1e-8))
print(f"base value {ex.base_value:+.3f} log-odds = {1 / (1 + np.exp(-ex.base_value)):.1%}")'''),
    md("""
The SHAP values are additive. For each employee, the base value plus the sum of the 51 contributions equals the model's
log-odds exactly. The feature-name mapping (one-hot columns back to raw attributes) is what the aggregation below
relies on.

## 2. Global importance
"""),
    code('''expl = shap.Explanation(values=ex.values, base_values=np.full(len(ex.values), ex.base_value),
                        data=ex.display_data, feature_names=ex.pretty_names)
shap.plots.bar(expl, max_display=15, show=False); plt.title("Mean |SHAP| per model input (log-odds)", loc="left"); plt.show()'''),
    code('''shap.plots.beeswarm(expl, max_display=15, show=False); plt.title("SHAP summary on the held-out set", loc="left"); plt.show()'''),
    md("""
**How to read the beeswarm.** Each dot is one test employee. Position shows how much that input moved their log-odds,
and colour shows the input's value (red is high). The model pushes risk **up** for overtime, years since last
promotion, number of previous employers, frequent travel and the Sales Representative and Laboratory Technician roles.
It pushes risk **down** for environment, job and relationship satisfaction, job involvement, total working years and
years with the current manager.

### Aggregated to the 30 raw attributes, compared with permutation importance
"""),
    code('''agg = aggregate_by_source(ex.values, ex.feature_names).abs().mean().sort_values(ascending=False)
perm = saved["permutation_importance"].set_index("feature")
comp = pd.DataFrame({"mean |SHAP|": agg, "SHAP rank": agg.rank(ascending=False).astype(int),
                     "permutation Δ PR-AUC": perm["importance_mean"], "perm SD": perm["importance_std"]})
comp["perm rank"] = comp["permutation Δ PR-AUC"].rank(ascending=False).astype(int)
comp = comp.sort_values("mean |SHAP|", ascending=False)
comp.index = [schema.PRETTY_NAMES.get(i, i) for i in comp.index]
from scipy.stats import spearmanr
print(f"Spearman rank agreement between the two methods: {spearmanr(comp['mean |SHAP|'], comp['permutation Δ PR-AUC']).statistic:.2f}")
comp.head(15)'''),
    code('''top = comp.head(12).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
axes[0].barh(top.index, top["mean |SHAP|"], color=F.TEAL); axes[0].set_title("mean |SHAP| (log-odds)")
axes[1].barh(top.index, top["permutation Δ PR-AUC"], xerr=top["perm SD"], color=F.BLUE, ecolor=F.MUTED); axes[1].set_title("permutation importance (PR-AUC drop)")
for ax in axes: ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.show()'''),
    md("""
**Agreement and disagreement**

* Both methods put **overtime** first, by a wide margin.
* Both rank job satisfaction, number of companies worked, years since promotion and environment satisfaction near the
  top.
* **Disagreements are informative.** Age, job role and the tenure counts get substantial SHAP values but small
  permutation importance. Shuffling one of a group of correlated features (age, total working years, tenure, level,
  income) costs little, because the others still carry the information. SHAP splits credit among them instead. Neither
  view is "the" importance.

## 3. Dependence: how one input's value maps to its effect
"""),
    code('''fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, feat in zip(axes, ["num__YearsSinceLastPromotion", "num__NumCompaniesWorked", "num__TotalWorkingYears"]):
    j = ex.feature_names.index(feat); yy = y_test.reindex(ex.row_ids).to_numpy()
    for v, c, lab in ((0, F.TEAL, "stayed"), (1, F.CLAY, "left")):
        ax.scatter(ex.display_data[yy == v, j], ex.values[yy == v, j], s=14, color=c, alpha=0.7, label=lab)
    ax.axhline(0, color=F.GRID); ax.set_title(schema.pretty_feature(feat)); ax.set_xlabel("value"); ax.set_ylabel("SHAP (log-odds)")
axes[0].legend(); plt.tight_layout(); plt.show()'''),
    md("""
For a logistic regression each dependence plot is a straight line by construction: the effect is linear in the
standardised feature. Real effects are probably not linear. Very long time since promotion is, for example, also a
mark of long tenure. This is a known limitation of the selected model, and the price of its stability on 1,176 training
rows.

## 4. Local explanations
"""),
    code('''tp = b.test_predictions.set_index("row_id")
hi = tp["probability"].idxmax(); lo = tp["probability"].idxmin()
for rid, title in ((hi, "highest predicted risk"), (lo, "lowest predicted risk")):
    i = int(np.flatnonzero(ex.row_ids == rid)[0])
    print(f"Employee #{int(tp.loc[rid, 'EmployeeNumber'])} ({title}): predicted {tp.loc[rid, 'probability']:.0%}, actually "
          f"{'left' if tp.loc[rid, 'actual_left'] else 'stayed'}")
    shap.plots.waterfall(expl[i], max_display=10, show=False); plt.title(title, loc="left"); plt.show()'''),
    md("""
The waterfall starts at the average log-odds and adds each input's contribution. For the highest-risk employee several
risk factors stack up: overtime, a sales or lab role, low satisfaction and frequent travel. For the lowest-risk employee,
long experience, high satisfaction and no overtime pull in the other direction.

These are explanations of a **score**. They do not establish that changing an input would change the outcome, and the
person behind the record is fictional.

## Summary

* SHAP values are exact for this linear model and reconstruct every prediction.
* The model's main associations are overtime, promotion recency, job hopping, satisfaction and career stage.
* Permutation importance agrees on the leaders but gives correlated career features less credit. Explanations of
  correlated inputs should be read as a group.
* None of this is causal evidence.
"""),
]

# =============================================================================================== 08
NB08 = [
    md("""
# 08 · Final pipeline, serialisation and deployment

How the selected model is packaged, loaded and served by the Streamlit app:

* **Reproducibility:** checks against the data hash, seeds, library versions and predictions.
* **Inference:** input validation and single-record prediction.
* **Speed:** timing for single records and batches.
* **Deployment:** how the app consumes the artefacts.
"""),
    code(SETUP),
    md("## 1. The artefacts on disk"),
    code('''import json, joblib
from talentscope.modeling import load_bundle
b = load_bundle()
for p in sorted((config.DEFAULT_EXPERIMENT_DIR).rglob("*")) + sorted(config.DEFAULT_MODEL_DIR.glob("*")):
    if p.is_file():
        print(f"{p.relative_to(config.ROOT).as_posix():55s} {p.stat().st_size / 1024:8.1f} KB")'''),
    code('''pd.Series({"selected": b.label, "params": b.selection["params"], "threshold": b.threshold,
           "created": b.metadata["created_utc"], "runtime (s)": b.metadata["runtime_seconds"]}).to_frame("value")'''),
    md("## 2. Reproducibility checks"),
    code('''from talentscope.data import dataset_fingerprint, load_dataset, split_features_target
from talentscope.preprocessing import make_split
import sklearn, imblearn, xgboost, lightgbm, shap
checks = {
    "dataset sha256 matches the run": dataset_fingerprint() == b.metadata["dataset"]["sha256"],
    "seed recorded": b.config["seed"] == config.RANDOM_SEED,
    "scikit-learn version matches": sklearn.__version__ == b.metadata["versions"]["scikit-learn"],
    "imbalanced-learn version matches": imblearn.__version__ == b.metadata["versions"]["imbalanced-learn"],
}
model = joblib.load(config.DEFAULT_MODEL_DIR / "model.joblib")
df = load_dataset(); X, y = split_features_target(df)
X_train, X_test, y_train, y_test = make_split(X, y, b.config["test_size"], b.config["seed"])
p = model.predict_proba(X_test)[:, 1]
checks["test row ids match saved predictions"] = list(X_test.index) == list(b.test_predictions["row_id"])
checks["loaded model reproduces saved probabilities"] = bool(np.allclose(p, b.test_predictions["probability"], atol=1e-6))
pd.Series(checks).to_frame("ok")'''),
    md("""
## 3. Round-trip serialisation

A model saved with `joblib` and reloaded must give identical outputs. The pipeline includes the imputer, the encoder
and the threshold-free model, so a single file is enough to serve raw records.
"""),
    code('''import tempfile, os
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "m.joblib"); joblib.dump(model, path, compress=3)
    again = joblib.load(path)
    print("identical after round trip:", np.array_equal(again.predict_proba(X_test), model.predict_proba(X_test)),
          f"| file size {os.path.getsize(path) / 1024:.1f} KB")
model'''),
    md("## 4. Inference on a single record, with validation"),
    code('''from talentscope.modeling.inference import typical_employee, validate_employee, predict_employee
rec = typical_employee(X_train)
res = predict_employee(model, rec, b.threshold)
print(f"typical employee: {res.probability:.1%} -> {res.band} (low < {res.low_cut:.0%} <= elevated < {res.threshold:.0%} <= high)")
risky = dict(rec, OverTime="Yes", JobSatisfaction=1, BusinessTravel="Travel_Frequently", YearsSinceLastPromotion=4,
             NumCompaniesWorked=6, MaritalStatus="Single")
r2 = predict_employee(model, risky, b.threshold)
print(f"same employee with several risk factors: {r2.probability:.1%} -> {r2.band}")'''),
    code('''bad = dict(rec, Age=22, TotalWorkingYears=15, YearsAtCompany=18, OverTime="Sometimes")
for e in validate_employee(bad):
    print("✗", e)
try:
    predict_employee(model, bad, b.threshold)
except ValueError as exc:
    print("\\nprediction refused:", type(exc).__name__)'''),
    md("## 5. Latency"),
    code('''import time
from talentscope.data import schema
from talentscope.modeling.inference import to_frame
one = to_frame(rec)
t = time.perf_counter()
for _ in range(200):
    model.predict_proba(one)
single = (time.perf_counter() - t) / 200
t = time.perf_counter(); model.predict_proba(X[schema.MODEL_FEATURES]); batch = time.perf_counter() - t
print(f"single record: {single * 1000:.2f} ms | all 1,470 employees: {batch * 1000:.1f} ms")'''),
    md("""
## 6. How the Streamlit app uses these artefacts

```
app.py                          st.navigation, brand bar, navigation rail
src/talentscope/ui/state.py     @st.cache_data     dataset(), split_frames()
                                @st.cache_resource load_bundle(run_dir), joblib model, SHAP explainer
app_pages/6_model_performance   reads cv_summary.csv, cv_folds.csv, oof_predictions.csv, test_*.{csv,json}
app_pages/7_explainable_ai      reads explain/shap_test.npz, permutation_importance.csv
app_pages/8_risk_simulator      model.joblib + validate_employee + LinearExplainer for the local waterfall
app_pages/5_model_laboratory    launches talentscope.modeling.run_job in a separate process -> experiments/runs/<id>
```

* No metric is typed into the UI. Every number on the model pages is read from these files on load.
* If no model exists, the pages show an empty state that points to `python scripts/train_default.py`.
* Lab runs are written next to the default run and appear in every page's experiment picker.

## 7. The model card
"""),
    code('''from talentscope.reporting import model_card
from talentscope.explainability import load_explanations
from IPython.display import Markdown
Markdown(model_card(b, load_explanations(b)))'''),
    md("""
## Summary

* One `joblib` file holds the full preprocessing and model pipeline; the threshold and metadata sit beside it in JSON.
* Reloading reproduces the saved test probabilities exactly, and the data hash and library versions match the run.
* Invalid records are rejected with readable reasons before scoring.
* Single-record inference takes on the order of 10 to 20 ms (mostly pandas overhead), fast enough for the interactive
  simulator; scoring all 1,470 employees takes about the same time.
"""),
]

NOTEBOOKS = {
    "01": ("01_Data_Understanding.ipynb", NB01),
    "02": ("02_Exploratory_Data_Analysis.ipynb", NB02),
    "03": ("03_Data_Preprocessing.ipynb", NB03),
    "04": ("04_Class_Imbalance_SMOTE.ipynb", NB04),
    "05": ("05_Model_Training_Comparison.ipynb", NB05),
    "06": ("06_Model_Evaluation.ipynb", NB06),
    "07": ("07_Explainable_AI_SHAP.ipynb", NB07),
    "08": ("08_Final_Pipeline_Deployment.ipynb", NB08),
}


def build(key: str, execute: bool = True) -> Path:
    name, cells = NOTEBOOKS[key]
    nb = nbformat.v4.new_notebook()
    nb.cells = [c for c in cells if c is not None]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    path = NB_DIR / name
    if execute:
        t0 = time.time()
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        print(f"executed {name} in {time.time() - t0:.0f} s", flush=True)
    nbformat.write(nb, path)
    return path


if __name__ == "__main__":
    keys = sys.argv[1:] or list(NOTEBOOKS)
    NB_DIR.mkdir(exist_ok=True)
    for k in keys:
        build(k)
