"""Project-wide constants: paths, seeds and the evaluation protocol."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ibm_hr_employee_attrition.csv"
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "default"
EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_EXPERIMENT_DIR = EXPERIMENTS_DIR / "default"
USER_EXPERIMENT_DIR = EXPERIMENTS_DIR / "runs"
ASSETS_DIR = ROOT / "assets"

# --- evaluation protocol -------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20              # stratified hold-out, split before anything is fitted
CV_FOLDS = 5                  # stratified folds for tuning, comparison and threshold selection
PRIMARY_METRIC = "average_precision"   # PR-AUC: informative for a 16% positive class
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# --- risk bands shown in the simulator ------------------------------------------
# Bands are relative to the selected decision threshold t, which is chosen on
# out-of-fold training predictions and never on the test set:
#   low       p <  0.5 * t
#   elevated  0.5 * t <= p < t
#   high      p >= t
RISK_LOW_FRACTION = 0.5

DATASET_NAME = "IBM HR Analytics Employee Attrition & Performance"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
DATASET_MIRROR_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
)
DATASET_LICENSE = (
    "Fictional dataset created by IBM data scientists; redistributed on Kaggle under the "
    "Open Database License (ODbL) and in IBM's public employee-attrition-aif360 repository"
)
REPO_URL = "https://github.com/SaiGrover/TalentScope"
