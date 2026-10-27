"""Diagnostics: constraint residual and drift, trajectory error, energy balance."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_simpson

from src.model import Control, Params, body_speed


def constraint_residual(X) -> np.ndarray:
    """r(X) = vx sin(theta) - vy cos(theta) [m/s] for X of shape (6,) or (6, N)."""
    X = np.asarray(X, dtype=float)
    if X.ndim not in (1, 2) or X.shape[0] != 6:
        raise ValueError(f"X must have shape (6,) or (6, N), got {X.shape}")
    return X[3] * np.sin(X[2]) - X[4] * np.cos(X[2])


def residual_drift(X) -> np.ndarray:
    """|r(t) - r(t0)| [m/s] along X (6, N). r is a first integral of the model, so this is integration error."""
    r = constraint_residual(X)
    if r.ndim != 1:
        raise ValueError("X must be a trajectory of shape (6, N)")
    return np.abs(r - r[0])


def position_error(X_a, X_b) -> np.ndarray:
    """Planar distance [m] between two trajectories (components x, y on axes 0, 1) at each time."""
    A, B = np.asarray(X_a, dtype=float), np.asarray(X_b, dtype=float)
    if A.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"incompatible trajectories: {A.shape} vs {B.shape}")
    return np.hypot(A[0] - B[0], A[1] - B[1])


def kinetic_energy(X, params: Params) -> np.ndarray:
    """E = m (vx^2 + vy^2) / 2 + J omega^2 / 2 [J]."""
    X = np.asarray(X, dtype=float)
    return 0.5 * params.m * (X[3] ** 2 + X[4] ** 2) + 0.5 * params.J * X[5] ** 2


def energy_balance_error(t, X, params: Params, control: Control) -> np.ndarray:
    """E(t) - E(t0) - int P ds [J], with P = F v + tau omega (Simpson quadrature, error O(h^4)).

    Zero in exact arithmetic when r = 0 (dE/dt = P - m v omega r in general).
    """
    t, X = np.asarray(t, dtype=float), np.asarray(X, dtype=float)
    F_tau = np.array([control(float(ti)) for ti in t])
    P = F_tau[:, 0] * body_speed(X) + F_tau[:, 1] * X[5]
    E = kinetic_energy(X, params)
    return E - E[0] - cumulative_simpson(P, x=t, initial=0.0)
