"""Baseline v1: planar rigid body with a no-lateral-slip constraint (provisional model).

Full state  X = (x, y, theta, vx, vy, omega)   [m, m, rad, m/s, m/s, rad/s]
Reduced     Z = (x, y, theta, v, omega),  v = vx cos(theta) + vy sin(theta)   (verification only)
Inputs      F [N] (body-axis force), tau [N m] (yaw torque); parameters m [kg], J [kg m^2].
Constraint  r(X) = vx sin(theta) - vy cos(theta) = 0.

    x'' = F/m cos(theta) - v omega sin(theta)
    y'' = F/m sin(theta) + v omega cos(theta)
    theta'' = tau / J

Derivation, assumptions and checks: notebooks/01_model_and_validation.ipynb.
Arrays have the component index on axis 0 (trajectory shape (n, N_times), as in solve_ivp).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Open-loop input: t [s] -> (F [N], tau [N m]).
Control = Callable[[float], tuple[float, float]]


@dataclass(frozen=True)
class Params:
    """Mass ``m`` [kg] and yaw inertia ``J`` [kg m^2]; both finite and > 0."""

    m: float
    J: float

    def __post_init__(self) -> None:
        for name in ("m", "J"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def constant_control(F: float, tau: float) -> Control:
    """Constant force F [N] and torque tau [N m]."""
    return lambda t: (F, tau)


def _check(a, n: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim not in (1, 2) or a.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},) or ({n}, N), got {a.shape}")
    return a


def body_speed(X) -> np.ndarray:
    """v = vx cos(theta) + vy sin(theta) for a full state (6,) or trajectory (6, N)."""
    X = _check(X, 6, "X")
    return X[3] * np.cos(X[2]) + X[4] * np.sin(X[2])


def rhs(t: float, X, params: Params, control: Control) -> np.ndarray:
    """dX/dt of the full 6-state model, shape (6,)."""
    X = _check(X, 6, "X")
    if X.ndim != 1:
        raise ValueError("rhs expects a single state of shape (6,)")
    F, tau = control(t)
    a = F / params.m
    theta, omega = X[2], X[5]
    c, s = math.cos(theta), math.sin(theta)
    v = X[3] * c + X[4] * s
    return np.array([X[3], X[4], omega, a * c - v * omega * s, a * s + v * omega * c, tau / params.J])


def rhs_reduced(t: float, Z, params: Params, control: Control) -> np.ndarray:
    """dZ/dt of the reduced 5-state model, shape (5,). Verification tool, not a second model."""
    Z = _check(Z, 5, "Z")
    if Z.ndim != 1:
        raise ValueError("rhs_reduced expects a single state of shape (5,)")
    F, tau = control(t)
    theta, v, omega = Z[2], Z[3], Z[4]
    return np.array([v * math.cos(theta), v * math.sin(theta), omega, F / params.m, tau / params.J])


def reconstruct(Z) -> np.ndarray:
    """Z (5, ...) -> X (6, ...) with vx = v cos(theta), vy = v sin(theta); gives r = 0."""
    x, y, theta, v, omega = _check(Z, 5, "Z")
    return np.stack([x, y, theta, v * np.cos(theta), v * np.sin(theta), omega])


def reduce_state(X) -> np.ndarray:
    """X (6, ...) -> Z (5, ...) by projecting the velocity on the body axis (drops lateral velocity)."""
    X = _check(X, 6, "X")
    return np.stack([X[0], X[1], X[2], body_speed(X), X[5]])


def free_motion_exact(t, x0: float, y0: float, theta0: float, v0: float, omega0: float) -> np.ndarray:
    """Closed-form X(t), shape (6, N), for F = tau = 0 and r(X0) = 0.

    omega0 != 0: circle of radius v0/omega0; omega0 == 0: straight line along theta0.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    th = theta0 + omega0 * t
    if omega0 == 0.0:
        x, y = x0 + v0 * np.cos(theta0) * t, y0 + v0 * np.sin(theta0) * t
    else:
        R = v0 / omega0
        x, y = x0 + R * (np.sin(th) - np.sin(theta0)), y0 - R * (np.cos(th) - np.cos(theta0))
    return np.stack([x, y, th, v0 * np.cos(th), v0 * np.sin(th), omega0 * np.ones_like(t)])
