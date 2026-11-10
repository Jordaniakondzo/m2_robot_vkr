"""Scenario definitions and runner shared by the KT3 and KT4 experiments."""

from __future__ import annotations

import numpy as np

from src.dataset import sample_targets
from src.metrics import constraint_residual, residual_drift
from src.model import Params, constant_control, free_motion_exact, reconstruct, reduce_state, rhs, rhs_reduced
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


def run_scenario(name: str, Z0, F: float, tau: float, closed_form) -> dict:
    """Integrate one scenario (6D and 5D models) on the dense grid; return trajectory, targets, diagnostics."""
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
    return {"name": name, "t": T, "X": X, "Z": Z, "targets": targets.X, "target_set": targets,
            "diagnostics": diag, "control": {"F": F, "tau": tau}}
