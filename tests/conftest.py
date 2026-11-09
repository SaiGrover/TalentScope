import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def raw():
    from talentscope.data import load_raw
    return load_raw()


@pytest.fixture(scope="session")
def df():
    from talentscope.data import load_dataset
    return load_dataset()


@pytest.fixture(scope="session")
def split(df):
    from talentscope.data import split_features_target
    from talentscope.preprocessing import make_split
    X, y = split_features_target(df)
    return make_split(X, y)


@pytest.fixture(scope="session")
def tiny_result():
    """A small but complete experiment, shared by the modelling tests."""
    from talentscope.modeling import ExperimentConfig, run_experiment
    cfg = ExperimentConfig(models=["dummy", "logreg", "tree"], strategies=["none", "smotenc"], n_iter=2,
                           cv_folds=3, cv_repeats=1, bootstrap_resamples=50, n_jobs=1)
    return run_experiment(cfg)
