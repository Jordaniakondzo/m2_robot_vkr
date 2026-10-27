"""KT3 experiment: three scenarios -> trajectories -> target states T_i = X(t_i) -> diagnostics -> figures.

    python -m experiments.run_kt3 [--out-root results]

Output: <out-root>/kt3_<UTC>/{targets.csv, metadata.json, paths.png, speeds.png, residuals.png}
(new directory, never overwritten). Synthetic parameters; the model is provisional.

S1 straight line and S2 circular arc (F = tau = 0) have closed forms (independent reference).
S3 has constant F and tau, so v and omega vary linearly in time; its reference is the 5-state model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.dataset import UNITS, code_version, sample_targets, save_targets
from src.metrics import constraint_residual, residual_drift
from src.model import Params, constant_control, free_motion_exact, reconstruct, reduce_state, rhs, rhs_reduced
from src.plots import plot_paths, plot_residuals, plot_speeds
from src.solver import integrate

PARAMS = Params(m=10.0, J=2.0)                           # kg, kg m^2
SOLVER = dict(method="DOP853", rtol=1e-10, atol=1e-12)
T = np.linspace(0.0, 10.0, 201)                          # dense grid, step 0.05 s
T_TARGETS = T[40::40]                                    # 2, 4, 6, 8, 10 s

# id, Z0 = (x, y, theta, v, omega), F [N], tau [N m], closed form (x0, y0, theta0, v0, omega0) or None
SCENARIOS = [
    ("S1_line", (0.0, 0.0, 0.0, 1.0, 0.0), 0.0, 0.0, (0.0, 0.0, 0.0, 1.0, 0.0)),
    ("S2_arc", (0.0, 0.0, 0.0, 1.5, 0.6), 0.0, 0.0, (0.0, 0.0, 0.0, 1.5, 0.6)),
    ("S3_const_F_tau", (0.0, 0.0, 0.0, 1.0, 0.3), 2.0, 0.2, None),
]


def run_scenario(name, Z0, F, tau, closed_form):
    """Integrate one scenario on the dense grid; return its trajectory, target set and diagnostics."""
    control = constant_control(F, tau)
    X0 = reconstruct(np.array(Z0))
    X = integrate(rhs, X0, T, PARAMS, control, **SOLVER)
    Z = integrate(rhs_reduced, reduce_state(X0), T, PARAMS, control, **SOLVER)
    diag = {"max_abs_r_dense": float(np.max(np.abs(constraint_residual(X)))),
            "max_drift_dense": float(np.max(residual_drift(X))),
            "max_state_diff_vs_5state_model": float(np.max(np.abs(reconstruct(Z) - X)))}
    if closed_form is not None:
        diag["max_state_error_vs_closed_form"] = float(np.max(np.abs(X - free_motion_exact(T, *closed_form))))
    meta = {"params": {"m": PARAMS.m, "J": PARAMS.J}, "initial_state_X0": X0.tolist(),
            "control": {"kind": "constant", "F": F, "tau": tau}, "solver": SOLVER,
            "t_targets": T_TARGETS.tolist(), "diagnostics": diag}
    targets = sample_targets(name, T, X, T_TARGETS, control, meta)
    return {"t": T, "X": X, "targets": targets.X, "set": targets}


def main(out_root="results") -> Path:
    runs = {name: run_scenario(name, *rest) for name, *rest in SCENARIOS}
    out = Path(out_root) / f"kt3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    save_targets([r["set"] for r in runs.values()], out, metadata={
        "model": "Baseline v1 (provisional), X = (x, y, theta, vx, vy, omega), r = vx sin(theta) - vy cos(theta)",
        "command": "python -m experiments.run_kt3", "units": UNITS,
        "note": "synthetic simulation output; not a measurement of a physical robot"})
    plot_paths(runs, out / "paths.png")
    plot_speeds(runs, out / "speeds.png")
    plot_residuals(runs, out / "residuals.png")
    for name, r in runs.items():
        print(name, r["set"].metadata["diagnostics"])
    print(f"written to {out} (commit {code_version()['commit']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default="results")
    main(ap.parse_args().out_root)
