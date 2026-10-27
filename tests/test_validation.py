"""Validation tests: closed forms, quadrature, energy balance and solver behaviour."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.metrics import constraint_residual, energy_balance_error, kinetic_energy, residual_drift
from src.model import Params, body_speed, constant_control, free_motion_exact, reconstruct, rhs
from src.solver import integrate

PARAMS = Params(m=10.0, J=2.0)
TOL_STATE = 1e-8
TOL_ENERGY = 1e-8  # includes the cumulative-Simpson error on the grid below
TIGHT = dict(method="DOP853", rtol=1e-12, atol=1e-14)
FREE = constant_control(0.0, 0.0)


def _full(X0, t, ctrl, **opts):
    return integrate(rhs, X0, t, PARAMS, ctrl, **(opts or TIGHT))


# --- closed forms --------------------------------------------------------------------------
@pytest.mark.parametrize("th0", [0.0, 0.7, -2.0])
def test_circle_matches_closed_form_and_closes(th0):
    t = np.linspace(0.0, 12.0, 241)
    x0, y0, v0, w0 = 1.0, -2.0, 1.5, 0.6
    X0 = reconstruct(np.array([x0, y0, th0, v0, w0]))
    np.testing.assert_allclose(_full(X0, t, FREE), free_motion_exact(t, x0, y0, th0, v0, w0), atol=TOL_STATE)
    period = 2 * np.pi / w0
    Xp = _full(X0, np.array([0.0, period]), FREE)
    np.testing.assert_allclose(Xp[:2, -1], [x0, y0], atol=TOL_STATE)


def test_straight_line():
    t = np.linspace(0.0, 5.0, 51)
    X0 = np.array([0.0, 0.0, 0.4, 2.0 * np.cos(0.4), 2.0 * np.sin(0.4), 0.0])
    X = _full(X0, t, FREE)
    np.testing.assert_allclose(X, free_motion_exact(t, 0.0, 0.0, 0.4, 2.0, 0.0), atol=TOL_STATE)
    assert np.max(np.abs(constraint_residual(X))) < TOL_STATE


def test_constant_force_and_torque_polynomials_plus_quadrature():
    """v, omega, theta are polynomial in t; x, y are the quadratures of v cos(theta), v sin(theta)."""
    F, tau, v0, w0 = 2.0, 0.2, 1.0, 0.3
    a, al = F / PARAMS.m, tau / PARAMS.J
    t = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    X = _full(reconstruct(np.array([0.0, 0.0, 0.0, v0, w0])), t, constant_control(F, tau))
    np.testing.assert_allclose(X[2], w0 * t + 0.5 * al * t**2, atol=TOL_STATE)
    np.testing.assert_allclose(X[5], w0 + al * t, atol=TOL_STATE)
    np.testing.assert_allclose(body_speed(X), v0 + a * t, atol=TOL_STATE)
    theta = lambda s: w0 * s + 0.5 * al * s**2
    for k, tk in enumerate(t):
        xq = quad(lambda s: (v0 + a * s) * np.cos(theta(s)), 0.0, tk, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
        yq = quad(lambda s: (v0 + a * s) * np.sin(theta(s)), 0.0, tk, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
        assert abs(X[0, k] - xq) < TOL_STATE and abs(X[1, k] - yq) < TOL_STATE


# --- energy --------------------------------------------------------------------------------
@pytest.mark.parametrize("ctrl", [constant_control(2.0, 0.2),
                                  lambda t: (3.0 * np.sin(0.7 * t), 0.4 * np.cos(1.3 * t) - 0.1)])
def test_energy_balance(ctrl):
    t = np.linspace(0.0, 10.0, 2001)
    X = _full(reconstruct(np.array([0.0, 0.0, 0.0, 1.0, 0.3])), t, ctrl)
    assert np.max(np.abs(energy_balance_error(t, X, PARAMS, ctrl))) < TOL_ENERGY
    assert np.ptp(kinetic_energy(X, PARAMS)) > 1e-2  # the balance is non-trivial


# --- solver --------------------------------------------------------------------------------
T = np.linspace(0.0, 12.0, 241)
Z0 = np.array([1.0, -2.0, 0.7, 1.5, 0.6])
EXACT = free_motion_exact(T, *Z0)


def _err(method, rtol):
    return np.max(np.abs(_full(reconstruct(Z0), T, FREE, method=method, rtol=rtol, atol=rtol * 1e-2) - EXACT))


def test_integrate_deterministic_and_shape():
    a, b = _full(reconstruct(Z0), T, FREE), _full(reconstruct(Z0), T, FREE)
    assert a.shape == (6, T.size) and np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("opts", [
    dict(method="DOP853", rtol=1e-15, atol=1e-9),      # scipy would silently clip it
    dict(method="DOP853", rtol=math.nan, atol=1e-9),
    dict(method="DOP853", rtol=1e-6, atol=-1.0),
])
def test_integrate_rejects_bad_tolerances(opts):
    with pytest.raises(ValueError):
        _full(reconstruct(Z0), T, FREE, **opts)


def test_integrate_rejects_bad_inputs_and_requires_tolerances():
    with pytest.raises(ValueError):
        _full(np.full(6, np.nan), T, FREE, **TIGHT)
    with pytest.raises(ValueError):
        _full(reconstruct(Z0), T[::-1], FREE, **TIGHT)
    with pytest.raises(TypeError):
        integrate(rhs, reconstruct(Z0), T, PARAMS, FREE)  # no default method / rtol / atol


def test_integrate_raises_on_failure():
    blow_up = lambda t, y, params, control: np.array([y[0] ** 2])  # blows up at t = 1
    with pytest.raises(RuntimeError):
        integrate(blow_up, [1.0], np.array([0.0, 2.0]), None, None, method="RK45", rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_error_decreases_with_tighter_tolerance(method):
    e3, e6, e9 = _err(method, 1e-3), _err(method, 1e-6), _err(method, 1e-9)
    assert e3 > 10 * e6 > 0 and e6 > 10 * e9


@pytest.mark.parametrize("method", ["RK23", "DOP853", "Radau", "BDF", "LSODA"])
def test_alternative_solvers_agree_with_rk45(method):
    X0 = reconstruct(Z0)
    ref = _full(X0, T, FREE, method="RK45", rtol=1e-9, atol=1e-11)
    got = _full(X0, T, FREE, method=method, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_loose_tolerance_makes_residual_drift_visible():
    """The integrator does not preserve r exactly; the drift is integration error, reported not hidden."""
    ctrl, X0 = constant_control(2.0, 0.2), reconstruct(np.array([0.0, 0.0, 0.0, 1.0, 0.3]))
    tight = _full(X0, T, ctrl)
    loose = _full(X0, T, ctrl, method="RK45", rtol=1e-3, atol=1e-6)
    assert residual_drift(loose).max() > 1e3 * residual_drift(tight).max()


def test_residual_drift_is_zero_for_constant_residual_whatever_r0():
    Xt = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.25, 0.3])[:, None], (1, 4))  # r0 = -0.25
    np.testing.assert_allclose(residual_drift(Xt), 0.0, atol=0)
