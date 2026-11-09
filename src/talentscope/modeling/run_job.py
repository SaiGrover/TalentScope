"""Run one experiment as a separate process (used by the Model Laboratory).

    python -m talentscope.modeling.run_job <run_dir>

``<run_dir>/request.json`` holds the ExperimentConfig. Progress is written to
``<run_dir>/progress.json`` after every step so the web app can poll it
without sharing a Python interpreter with the training code. A run directory
only becomes visible to the app once ``selection.json`` and ``model.joblib``
exist, so a crashed run never shows up as a result.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def _write(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)


def main(run_dir: str) -> int:
    run_dir = Path(run_dir)
    progress_path = run_dir / "progress.json"
    state = {"status": "running", "progress": 0.0, "message": "Loading data", "log": [], "error": "",
             "started": datetime.now().isoformat(timespec="seconds"), "finished": None}
    _write(progress_path, state)

    def progress(f: float, msg: str, scale: float = 0.9) -> None:
        state["progress"] = round(scale * f, 4)
        state["message"] = msg
        state["log"].append(f"{datetime.now():%H:%M:%S}  {msg}")
        _write(progress_path, state)

    try:
        from talentscope.data import load_dataset, split_features_target
        from talentscope.explainability import explain_bundle
        from talentscope.modeling import ExperimentConfig, load_bundle, run_experiment, save_bundle
        from talentscope.preprocessing import make_split

        cfg = ExperimentConfig(**json.loads((run_dir / "request.json").read_text(encoding="utf-8")))
        t0 = time.time()
        result = run_experiment(cfg, progress)
        save_bundle(result, run_dir)
        progress(1.0, "Computing explanations (SHAP, permutation importance, group metrics)")
        df = load_dataset()
        X, y = split_features_target(df)
        X_train, X_test, _, y_test = make_split(X, y, cfg.test_size, cfg.seed)
        explain_bundle(load_bundle(run_dir), X_train, X_test, y_test,
                       groups=df[["Gender", "AgeBand", "MaritalStatus", "Department"]],
                       kernel_nsamples=150, n_repeats=10)
        state.update(status="done", progress=1.0, message=f"Finished in {time.time() - t0:.0f} s")
        state["log"].append(f"{datetime.now():%H:%M:%S}  {state['message']}")
        return_code = 0
    except Exception as exc:  # reported to the page, never swallowed
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        state["log"].append(traceback.format_exc())
        return_code = 1
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _write(progress_path, state)
    return return_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
