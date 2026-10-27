"""Numerical integration: a checked wrapper around ``scipy.integrate.solve_ivp``.

``method``, ``rtol`` and ``atol`` are required keyword arguments (no hidden defaults).
Errors are raised, never returned as partial solutions.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

MIN_RTOL = 100 * np.finfo(float).eps  # below this scipy silently raises rtol


def integrate(rhs, y0, t_eval, params, control, *, method: str, rtol: float, atol: float) -> np.ndarray:
    """Integrate ``y' = rhs(t, y, params, control)`` and return y sampled at ``t_eval``, shape (n, N).

    ``t_eval`` is finite and strictly increasing; the interval is [t_eval[0], t_eval[-1]].
    Raises ValueError on invalid input and RuntimeError on solver failure or non-finite output.
    """
    y0, t_eval = np.asarray(y0, dtype=float), np.asarray(t_eval, dtype=float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise ValueError("y0 must be a finite 1-D array")
    if t_eval.ndim != 1 or t_eval.size < 2 or not np.all(np.isfinite(t_eval)) or np.any(np.diff(t_eval) <= 0):
        raise ValueError("t_eval must be finite, strictly increasing, with >= 2 points")
    if not rtol >= MIN_RTOL:
        raise ValueError(f"rtol must be >= {MIN_RTOL:.3g}, got {rtol!r}")
    if not atol >= 0.0:
        raise ValueError(f"atol must be >= 0, got {atol!r}")
    sol = solve_ivp(lambda t, y: rhs(t, y, params, control), (t_eval[0], t_eval[-1]), y0,
                    method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed ({method}): {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError("integration returned non-finite values")
    return sol.y
