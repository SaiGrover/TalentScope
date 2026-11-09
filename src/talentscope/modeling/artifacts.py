"""Persist and reload experiment results.

A run directory holds everything needed to reproduce and audit a result::

    config.json             ExperimentConfig, exactly as run
    metadata.json           dataset hash, split, protocol, library versions, runtime
    selection.json          selected model, strategy, parameters, threshold and why
    cv_summary.csv          one row per (model, strategy): CV mean/std, OOF metrics
    cv_folds.csv            fold-level metrics for every configuration
    search_trials.csv       every hyperparameter candidate tried
    best_params.json        tuned parameters per configuration
    oof_predictions.csv     out-of-fold training probabilities per configuration
    test_predictions.csv    selected model's probabilities on the untouched test set
    test_metrics.json       held-out metrics at the selected threshold
    test_bootstrap.csv      95% bootstrap intervals on the held-out metrics
    test_curves.json        ROC / PR curve points on the test set
    calibration_*.csv       reliability tables
    classification_report.csv
    resampling.json         class counts before/after resampling in a training fold
    model.joblib            the fitted pipeline (impute -> [sampler] -> encode -> model)
    explain/                SHAP values, permutation importance, group metrics

The default run is committed under ``experiments/default`` with its model in
``models/default``; runs launched from the app go to ``experiments/runs``.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from talentscope import config


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_bundle(result, run_dir: Path, model_dir: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", result.config)
    write_json(run_dir / "metadata.json", result.metadata)
    write_json(run_dir / "selection.json", result.selection)
    write_json(run_dir / "best_params.json", result.best_params)
    write_json(run_dir / "test_metrics.json", result.test_metrics)
    write_json(run_dir / "test_curves.json", result.test_curves)
    write_json(run_dir / "resampling.json", result.resampling)
    result.cv_summary.round(6).to_csv(run_dir / "cv_summary.csv", index=False)
    result.cv_folds.round(6).to_csv(run_dir / "cv_folds.csv", index=False)
    result.search_trials.round(6).to_csv(run_dir / "search_trials.csv", index=False)
    result.oof.round(6).to_csv(run_dir / "oof_predictions.csv")
    result.test_predictions.round(6).to_csv(run_dir / "test_predictions.csv", index=False)
    result.test_bootstrap.round(6).to_csv(run_dir / "test_bootstrap.csv", index=False)
    result.test_calibration.round(6).to_csv(run_dir / "calibration_test.csv", index=False)
    result.oof_calibration.round(6).to_csv(run_dir / "calibration_oof.csv", index=False)
    result.classification_report.round(6).to_csv(run_dir / "classification_report.csv", index=False)
    model_dir = Path(model_dir) if model_dir is not None else run_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_dir / "model.joblib", compress=3)
    if model_dir != run_dir:
        write_json(model_dir / "model_metadata.json", {
            "experiment_dir": run_dir.resolve().relative_to(config.ROOT).as_posix()
            if run_dir.resolve().is_relative_to(config.ROOT) else str(run_dir),
            "selection": result.selection, "test_metrics": result.test_metrics,
            "versions": result.metadata["versions"], "dataset_sha256": result.metadata["dataset"]["sha256"],
            "created_utc": result.metadata["created_utc"],
        })
    return run_dir


@dataclass
class ModelBundle:
    run_dir: Path
    model_path: Path
    config: dict
    metadata: dict
    selection: dict
    best_params: dict
    cv_summary: pd.DataFrame
    cv_folds: pd.DataFrame
    search_trials: pd.DataFrame
    oof: pd.DataFrame
    test_predictions: pd.DataFrame
    test_metrics: dict
    test_bootstrap: pd.DataFrame
    test_curves: dict
    test_calibration: pd.DataFrame
    oof_calibration: pd.DataFrame
    classification_report: pd.DataFrame
    resampling: dict

    @property
    def threshold(self) -> float:
        return float(self.selection["threshold"])

    @property
    def label(self) -> str:
        return f"{self.selection['model_name']} + {self.selection['strategy_name']}"

    def load_model(self):
        return joblib.load(self.model_path)

    @property
    def explain_dir(self) -> Path:
        return self.run_dir / "explain"


def load_bundle(run_dir: Path | None = None, model_path: Path | None = None) -> ModelBundle:
    run_dir = Path(run_dir) if run_dir is not None else config.DEFAULT_EXPERIMENT_DIR
    if model_path is None:
        candidate = run_dir / "model.joblib"
        model_path = candidate if candidate.exists() else config.DEFAULT_MODEL_DIR / "model.joblib"
    if not (run_dir / "selection.json").exists():
        raise FileNotFoundError(f"No experiment found in {run_dir}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model artefact missing: {model_path}")
    csv = lambda name, **kw: pd.read_csv(run_dir / name, **kw) if (run_dir / name).exists() else pd.DataFrame()
    return ModelBundle(
        run_dir=run_dir, model_path=Path(model_path),
        config=read_json(run_dir / "config.json"), metadata=read_json(run_dir / "metadata.json"),
        selection=read_json(run_dir / "selection.json"), best_params=read_json(run_dir / "best_params.json"),
        cv_summary=csv("cv_summary.csv"), cv_folds=csv("cv_folds.csv"), search_trials=csv("search_trials.csv"),
        oof=csv("oof_predictions.csv", index_col="row_id"), test_predictions=csv("test_predictions.csv"),
        test_metrics=read_json(run_dir / "test_metrics.json"), test_bootstrap=csv("test_bootstrap.csv"),
        test_curves=read_json(run_dir / "test_curves.json"), test_calibration=csv("calibration_test.csv"),
        oof_calibration=csv("calibration_oof.csv"), classification_report=csv("classification_report.csv"),
        resampling=read_json(run_dir / "resampling.json"),
    )


def list_runs() -> list[Path]:
    """Default run first, then app-launched runs newest first."""
    runs = []
    if (config.DEFAULT_EXPERIMENT_DIR / "selection.json").exists():
        runs.append(config.DEFAULT_EXPERIMENT_DIR)
    if config.USER_EXPERIMENT_DIR.exists():
        user = [p for p in config.USER_EXPERIMENT_DIR.iterdir() if (p / "selection.json").exists()
                and (p / "model.joblib").exists()]
        runs.extend(sorted(user, key=lambda p: p.name, reverse=True))
    return runs


def delete_run(run_dir: Path) -> None:
    run_dir = Path(run_dir).resolve()
    if config.USER_EXPERIMENT_DIR.resolve() in run_dir.parents:
        shutil.rmtree(run_dir)
