"""Background training jobs for the Model Laboratory.

Each experiment runs in its own Python process (``talentscope.modeling.run_job``)
so model fitting never competes with the web server for the interpreter lock:
the dashboard stays responsive and the user can keep exploring other pages.
The process reports progress through ``progress.json`` in its run directory,
which the lab page polls with a fragment. Nothing is retrained on a rerun,
because the process, not the page script, owns the work.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st

from talentscope import config


@dataclass
class Job:
    id: str
    run_dir: Path
    status: str = "running"            # running | done | failed
    progress: float = 0.0
    message: str = "Starting"
    log: list[str] = field(default_factory=list)
    error: str = ""


@st.cache_resource
def _registry() -> dict:
    """Process handles survive reruns and are shared by every session on this server."""
    return {"procs": {}, "lock": threading.Lock()}


def _read(job_id: str) -> Job | None:
    run_dir = config.USER_EXPERIMENT_DIR / job_id
    p = run_dir / "progress.json"
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Job(id=job_id, run_dir=run_dir)
    job = Job(id=job_id, run_dir=run_dir, status=s.get("status", "running"), progress=s.get("progress", 0.0),
              message=s.get("message", ""), log=s.get("log", []), error=s.get("error", ""))
    proc = _registry()["procs"].get(job_id)
    if job.status == "running" and proc is not None and proc.poll() not in (None, 0):
        job.status = "failed"
        job.error = job.error or f"Training process exited with code {proc.returncode}; see job.log."
    return job


def submit(cfg) -> Job:
    reg = _registry()
    with reg["lock"]:
        if any(p.poll() is None for p in reg["procs"].values()):
            raise RuntimeError("Another experiment is already running on this server. Wait for it to finish.")
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = config.USER_EXPERIMENT_DIR / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "request.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        (run_dir / "progress.json").write_text(json.dumps({"status": "running", "progress": 0.0,
                                                           "message": "Starting", "log": []}), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(config.ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "ignore"
        log = open(run_dir / "job.log", "w", encoding="utf-8")  # noqa: SIM115 - handed to the child
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        reg["procs"][job_id] = subprocess.Popen(
            [sys.executable, "-m", "talentscope.modeling.run_job", str(run_dir)],
            cwd=str(config.ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return _read(job_id)


def get(job_id: str | None) -> Job | None:
    return _read(job_id) if job_id else None


def any_running() -> Job | None:
    for job_id, proc in _registry()["procs"].items():
        if proc.poll() is None:
            return _read(job_id)
    return None


def latest() -> Job | None:
    """The running job if there is one, else the most recently started job on this server."""
    running = any_running()
    if running is not None:
        return running
    ids = sorted(_registry()["procs"])
    return _read(ids[-1]) if ids else None
