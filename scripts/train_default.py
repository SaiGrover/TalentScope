"""Reproduce the committed default experiment and model.

    python scripts/train_default.py            # full protocol (about 15-25 minutes on a laptop)
    python scripts/train_default.py --quick    # tiny search, for checking the pipeline only

Writes ``experiments/default`` (all evidence) and ``models/default/model.joblib``.
Every number the app shows for the default model is read from these files.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talentscope import config  # noqa: E402
from talentscope.data import load_dataset, split_features_target  # noqa: E402
from talentscope.explainability import explain_bundle  # noqa: E402
from talentscope.modeling import ExperimentConfig, load_bundle, run_experiment, save_bundle  # noqa: E402
from talentscope.preprocessing import make_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(config.DEFAULT_EXPERIMENT_DIR))
    ap.add_argument("--model-dir", default=str(config.DEFAULT_MODEL_DIR))
    args = ap.parse_args()

    cfg = ExperimentConfig(name="default")
    if args.quick:
        cfg.n_iter, cfg.cv_repeats, cfg.bootstrap_resamples = 3, 1, 200
    start = time.time()

    def progress(f, msg):
        print(f"[{time.time() - start:7.1f}s] {f:5.1%}  {msg}", flush=True)

    result = run_experiment(cfg, progress)
    run_dir = save_bundle(result, Path(args.out), Path(args.model_dir))
    print("selected:", result.selection["model_name"], "+", result.selection["strategy_name"],
          "threshold", result.selection["threshold"])
    print("test:", {k: round(v, 4) for k, v in result.test_metrics.items() if isinstance(v, float)})

    df = load_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_split(X, y, cfg.test_size, cfg.seed)
    bundle = load_bundle(run_dir, Path(args.model_dir) / "model.joblib")
    progress(1.0, "Computing SHAP values, permutation importance and group metrics")
    groups = df[["Gender", "AgeBand", "MaritalStatus", "Department"]]
    explain_bundle(bundle, X_train, X_test, y_test, groups=groups,
                   kernel_nsamples=200 if args.quick else 400, n_repeats=5 if args.quick else 20)
    progress(1.0, f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
