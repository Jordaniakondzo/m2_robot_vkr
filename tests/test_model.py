"""Model: shapes, input checks, analytic identities of the constraint, full vs reduced equivalence.
Numerical thresholds are technical software-validation thresholds, not physical tolerances (not validated by the
supervisor). They exceed by at least a factor 40 the errors observed with DOP853, rtol=1e-12, atol=1e-14
(states <= 4e-11, energy balance <= 2.3e-10 including the Simpson error); see README, 'Test tolerances'.
"""

import numpy as np
import pytest
import sympy as sp

from src.metrics import constraint_residual, position_error
from src.model import (Params, body_speed, constant_control, reconstruct, reduce_state, rhs,
                       rhs_reduced)
from src.solver import integrate

PARAMS = Params(m=10.0, J=2.0)
CTRL = constant_control(2.0, 0.2)
TIME_VARYING = lambda t: (3.0 * np.sin(0.7 * t), 0.4 * np.cos(1.3 * t) - 0.1)
EPS = 1e-12
RNG_SEED = 20261024


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


# --- shapes, determinism, validation ------------------------------------------------------
def test_rhs_shapes_finite_deterministic():
    X0, Z0 = np.array([0, 0, 0.3, 1, 0.2, 0.3]),np.array([0, 0, 0.3, 1, 0.3])
    f, g = rhs(0.0, X0, PARAMS, CTRL), rhs_reduced(0.0, Z0, PARAMS, CTRL)
    assert f.shape == (6,) and g.shape == (5,) and np.all(np.isfinite(f)) and np.all(np.isfinite(g))
    np.testing.assert_array_equal(f, rhs(0.0, list(X0), PARAMS, CTRL))


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(7), np.zeros((6, 2)), np.zeros(())])
def test_rhs_rejects_bad_shape(bad):
    with pytest.raises(ValueError):
        rhs(0.0, bad, PARAMS, CTRL)


@pytest.mark.parametrize("m,J", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (np.nan, 1.0), (1.0, np.inf)])
def test_params_validation(m, J):
    with pytest.raises(ValueError):
        Params(m=m, J=J)


def test_trajectory_shapes():
    Z = np.tile(np.array([0, 0, 0.3, 1, 0.3])[:, None], (1, 7))
    assert reconstruct(Z).shape == (6, 7) and reduce_state(reconstruct(Z)).shape == (5, 7)
    assert body_speed(reconstruct(Z)).shape == (7,)
    with pytest.raises(ValueError):
        reconstruct(np.zeros(6))


# --- reconstruction -----------------------------------------------------------------------
def test_reconstruction_is_admissible_and_invertible(rng):
    Z = rng.uniform(-3, 3, size=(5, 200))
    Z[2] = rng.uniform(-10, 10, size=200)
    X = reconstruct(Z)
    assert np.max(np.abs(constraint_residual(X))) < 1e-13
    np.testing.assert_allclose(reduce_state(X), Z, atol=1e-13, rtol=0)


def test_residual_is_minus_body_lateral_velocity(rng):
    X = rng.uniform(-2, 2, size=(6, 50))
    lateral = -X[3] * np.sin(X[2]) + X[4] * np.cos(X[2])
    np.testing.assert_allclose(constraint_residual(X), -lateral, atol=1e-13)


# --- analytic identities: r' = 0, v' = F/m - omega r, lambda = -m v omega ----------------
def test_sympy_reaction_force_and_invariance():
    """Independent derivation from Lagrange-d'Alembert (does not use src)."""
    m, F, lam = sp.symbols("m F lambda", real=True)
    th, vx, vy, w = sp.symbols("theta v_x v_y omega", real=True)
    v = vx * sp.cos(th) + vy * sp.sin(th)
    r = vx * sp.sin(th) - vy * sp.cos(th)
    ax = (F * sp.cos(th) + lam * sp.sin(th)) / m  # M q'' = B u + A^T lambda, A = [sin, -cos, 0]
    ay = (F * sp.sin(th) - lam * sp.cos(th)) / m

    def rdot(ax, ay):
        return ax * sp.sin(th) + w * vx * sp.cos(th) - ay * sp.cos(th) + w * vy * sp.sin(th)

    (lam_sol,) = sp.solve(sp.Eq(sp.simplify(rdot(ax, ay)), 0), lam)
    assert sp.simplify(lam_sol + m * v * w) == 0
    ax_s, ay_s = ax.subs(lam, -m * v * w), ay.subs(lam, -m * v * w)
    assert sp.simplify(rdot(ax_s, ay_s)) == 0  # r' = 0 for ANY (theta, vx, vy)
    vdot = ax_s * sp.cos(th) + ay_s * sp.sin(th) + w * (-vx * sp.sin(th) + vy * sp.cos(th))
    assert sp.simplify(vdot - (F / m - w * r)) == 0


def test_implementation_identities_off_constraint(rng):
    """r' = 0, v' = F/m - omega r and the reaction form, at random states including r != 0."""
    F, tau = 3.7, -1.1
    ctrl = constant_control(F, tau)
    for X in rng.uniform(-3, 3, size=(300, 6)):
        th, w = X[2], X[5]
        f = rhs(0.0, X, PARAMS, ctrl)
        scale = EPS * (1 + np.abs(X).max() ** 2)
        rdot = f[3] * np.sin(th) - f[4] * np.cos(th) + w * body_speed(X)
        vdot = f[3] * np.cos(th) + f[4] * np.sin(th) + w * (-X[3] * np.sin(th) + X[4] * np.cos(th))
        assert abs(rdot) < scale
        assert abs(vdot - (F / PARAMS.m - w * constraint_residual(X))) < scale
        lam = -PARAMS.m * body_speed(X) * w
        assert abs(PARAMS.m * f[3] - (F * np.cos(th) + lam * np.sin(th))) < scale
        assert abs(PARAMS.m * f[4] - (F * np.sin(th) - lam * np.cos(th))) < scale


# --- full (6) vs reduced (5) formulation --------------------------------------------------
def test_vector_fields_agree_on_admissible_set(rng):
    """rhs(R(Z)) equals the chain-rule derivative of R(Z) built from rhs_reduced."""
    for Z in rng.uniform(-3, 3, size=(100, 5)):
        th, v, w = Z[2], Z[3], Z[4]
        dZ = rhs_reduced(0.4, Z, PARAMS, TIME_VARYING)
        chain = np.array([dZ[0], dZ[1], dZ[2], dZ[3] * np.cos(th) - v * np.sin(th) * w,
                          dZ[3] * np.sin(th) + v * np.cos(th) * w, dZ[4]])
        np.testing.assert_allclose(rhs(0.4, reconstruct(Z), PARAMS, TIME_VARYING), chain, atol=EPS)


def _solve(fun, y0, t, ctrl):
    return integrate(fun, y0, t, PARAMS, ctrl, method="DOP853", rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("ctrl", [CTRL, TIME_VARYING, constant_control(0.0, 0.0)])
def test_trajectories_equivalent_on_admissible_set(ctrl):
    t, Z0 = np.linspace(0.0, 10.0, 201), np.array([0.0, 0.0, 0.0, 1.0, 0.3])
    X, Z = _solve(rhs, reconstruct(Z0), t, ctrl), _solve(rhs_reduced, Z0, t, ctrl)
    assert np.max(np.abs(X - reconstruct(Z))) < 1e-8
    assert position_error(X, Z).max() < 1e-8
    assert np.max(np.abs(constraint_residual(X))) < 1e-9


def test_residual_is_conserved_off_constraint_and_formulations_differ():
    """r is a first integral (neither grows nor decays), so the 5-state model no longer applies when r0 != 0."""
    t = np.linspace(0.0, 10.0, 201)
    X0 = np.array([0.0, 0.0, 0.3, 1.0, 0.4, 0.3])
    r0 = float(constraint_residual(X0))
    assert abs(r0) > 0.05
    X = _solve(rhs, X0, t, CTRL)
    assert np.max(np.abs(constraint_residual(X) - r0)) < 1e-9
    Z = _solve(rhs_reduced, reduce_state(X0), t, CTRL)
    assert position_error(X, Z).max() > 1e-2
