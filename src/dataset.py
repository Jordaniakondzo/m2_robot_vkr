"""Target states T_i = X(t_i) from a simulated trajectory; CSV + JSON export and reload.

Numbers are written with repr() precision, so save/load is bit-exact. A directory is never overwritten.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from src.metrics import constraint_residual
from src.model import Control

COLUMNS = ("trajectory_id", "t", "x", "y", "theta", "vx", "vy", "omega", "r", "F", "tau")
R_STORED_ATOL = 1e-12  # [m/s] stored r vs r recomputed from the stored state (exact up to repr round trip)
UNITS = {"t": "s", "x": "m", "y": "m", "theta": "rad", "vx": "m/s", "vy": "m/s", "omega": "rad/s",
         "r": "m/s", "F": "N", "tau": "N*m"}


@dataclass(frozen=True)
class TargetSet:
    """Targets of one trajectory: ``t`` (N,), ``X`` (6, N), ``inputs`` = (F, tau) at t_i, shape (2, N)."""

    trajectory_id: str
    t: np.ndarray
    X: np.ndarray
    inputs: np.ndarray
    metadata: dict = field(default_factory=dict)


def sample_targets(trajectory_id: str, t, X, t_targets, control: Control, metadata: dict | None = None) -> TargetSet:
    """Pick the columns of the trajectory (t, X) at ``t_targets``, which must be grid points of ``t``."""
    t, X, t_targets = (np.asarray(a, dtype=float) for a in (t, X, t_targets))
    idx = np.searchsorted(t, t_targets)
    if np.any(idx >= t.size) or not np.allclose(t[idx], t_targets, rtol=0, atol=1e-12):
        raise ValueError("every target time must be a grid point of t")
    inputs = np.array([control(float(ti)) for ti in t[idx]]).T
    return TargetSet(trajectory_id, t[idx].copy(), X[:, idx].copy(), inputs, dict(metadata or {}))


def code_version() -> dict:
    """Git commit and dirty flag of the working tree (None if git is unavailable)."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def save_targets(target_sets: list[TargetSet], out_dir, metadata: dict | None = None) -> Path:
    """Write ``targets.csv`` and ``metadata.json`` into a NEW directory (FileExistsError otherwise)."""
    ids = [ts.trajectory_id for ts in target_sets]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("need at least one target set, with unique trajectory ids")
    for ts in target_sets:
        n = ts.t.size
        if ts.X.shape != (6, n) or ts.inputs.shape != (2, n) or not np.all(np.isfinite(ts.X)):
            raise ValueError(f"{ts.trajectory_id}: inconsistent or non-finite target set")
    code = code_version()  # before mkdir, so the new directory does not make the tree dirty
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "targets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for ts in target_sets:
            r = constraint_residual(ts.X)
            for i in range(ts.t.size):
                row = [ts.t[i], *ts.X[:, i], r[i], *ts.inputs[:, i]]
                w.writerow([ts.trajectory_id, *(repr(float(v)) for v in row)])
    meta = {"created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "columns": list(COLUMNS),
            "units": UNITS, "code": code,
            "software": {"python_version": platform.python_version(), "numpy_version": np.__version__,
                         "scipy_version": scipy.__version__},
            "dataset": dict(metadata or {}),
            "trajectories": {ts.trajectory_id: ts.metadata for ts in target_sets}}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
    return out


def load_targets(directory) -> list[TargetSet]:
    """Load a dataset written by ``save_targets``; ValueError if the schema does not match."""
    d = Path(directory)
    meta = json.loads((d / "metadata.json").read_text())
    with open(d / "targets.csv", newline="") as f:
        reader = csv.reader(f)
        if tuple(next(reader)) != COLUMNS:
            raise ValueError("unexpected CSV columns")
        rows = list(reader)
    blocks: dict[str, list] = {}
    for row in rows:
        if len(row) != len(COLUMNS):
            raise ValueError(f"malformed CSV row: {row}")
        blocks.setdefault(row[0], []).append([float(v) for v in row[1:]])
    if set(blocks) != set(meta["trajectories"]):
        raise ValueError("trajectory ids differ between CSV and metadata")
    out = []
    for tid, block in blocks.items():
        a = np.array(block)  # columns: t, 6 states, r, F, tau
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{tid}: non-finite values")
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError(f"{tid}: times must be strictly increasing")
        if not np.allclose(a[:, 7], constraint_residual(a[:, 1:7].T), rtol=0, atol=R_STORED_ATOL):
            raise ValueError(f"{tid}: stored r inconsistent with the stored state")
        out.append(TargetSet(tid, a[:, 0].copy(), a[:, 1:7].T.copy(), a[:, 8:10].T.copy(), meta["trajectories"][tid]))
    return out
