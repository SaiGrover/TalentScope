# TalentScope notebooks

Eight executed notebooks document the data science and machine learning workflow behind the app. They import the
same `talentscope` package as the Streamlit app (`src/talentscope`), so the notebooks show the code that runs, not a
re-implementation. They use the same split (stratified 80/20, seed 42), the same pipelines and the same saved
experiment in `experiments/default`.

| # | Notebook | Purpose | Runtime |
|---|---|---|---|
| 01 | [Data understanding](01_Data_Understanding.ipynb) | Source and version hash, schema and roles, target distribution, missing values, duplicates, constant and redundant columns, unexpected values, leakage screen | seconds |
| 02 | [Exploratory data analysis](02_Exploratory_Data_Analysis.ipynb) | Univariate distributions; attrition by 12 dimensions with Wilson intervals; chi-square and Mann-Whitney tests with Benjamini-Hochberg correction; correlation and Cramér's V; role × overtime interaction | seconds |
| 03 | [Data preprocessing](03_Data_Preprocessing.ipynb) | Stratified split, the impute → resample → encode → model pipeline, training-only fitting, missing and unseen values, SMOTENC rounding, and a **leakage experiment** showing what resampling before CV does to the score | seconds |
| 04 | [Class imbalance and SMOTE](04_Class_Imbalance_SMOTE.ipynb) | What imbalance does to naive models, how SMOTENC works (PCA view of a resampled fold), the controlled comparison of four strategies, and a SMOTENC sensitivity grid. Reports the **negative result**: SMOTENC did not improve PR-AUC | ~20 s |
| 05 | [Model training and comparison](05_Model_Training_Comparison.ipynb) | **Re-runs the full default experiment** (7 algorithms × 4 strategies, randomised search, repeated CV), asserts it matches the committed artefacts, and analyses rankings, fold variability, paired differences and the search | ~10 min |
| 06 | [Model evaluation](06_Model_Evaluation.ipynb) | Threshold chosen on out-of-fold predictions; held-out metrics with bootstrap intervals; ROC and PR curves, confusion matrix, calibration, learning curve, error rates by group | ~30 s |
| 07 | [Explainable AI with SHAP](07_Explainable_AI_SHAP.ipynb) | SHAP recomputed and checked for additivity; global bar and beeswarm; SHAP versus permutation importance; dependence plots; local waterfalls for the highest- and lowest-risk employees | ~20 s |
| 08 | [Final pipeline and deployment](08_Final_Pipeline_Deployment.ipynb) | Artefact inventory, reproducibility checks (data hash, versions, predictions), joblib round trip, validated single-record inference, latency, how the app consumes the artefacts, the model card | seconds |

## Running them

Run from the repository root.

```bash
pip install -r requirements-dev.txt
```

```bash
python scripts/build_notebooks.py
```

The build script writes each notebook from its cell definitions and executes it top to bottom, so committed outputs
are always from a full, clean run. To rebuild selected notebooks only:

```bash
python scripts/build_notebooks.py 03 07
```

You can also open the notebooks in Jupyter and run them directly. The first cell adds `src/` to the import path.

## Conventions

* Static matplotlib figures (GitHub renders PNG outputs). Teal means *stayed*, clay means *left*, as in the app.
* Rates come with group sizes and 95% Wilson intervals. Tests report Benjamini-Hochberg adjusted p-values. Held-out
  metrics come with 95% bootstrap intervals.
* Unsuccessful experiments (SMOTENC, random forest, the single decision tree) are reported, not dropped.
* The data is fictional. Findings are associations within it.
