"""Geometry and figure smoke tests for the planar-motion views (no solver involved)."""

import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.animation import FuncAnimation

from src.plots import animate_planar_motion, plot_planar_motion, robot_shape, transform_robot_pose


def synthetic_run(n=21):
    t = np.linspace(0.0, 2.0, n)
    X = np.zeros((6, n))
    X[0], X[2] = t, 0.3 * t
    X[1] = 0.1 * t ** 2
    return {"name": "synthetic", "t": t, "X": X, "targets": X[:, [10, 20]]}


def test_robot_shape_has_expected_dimensions():
    shape = robot_shape()
    assert shape.shape == (2, 5)
    assert np.isfinite(shape).all()
    np.testing.assert_array_equal(shape[:, 0], shape[:, -1])  # closed


@pytest.mark.parametrize("bad", [0.0, -0.1, np.nan, np.inf])
def test_robot_shape_rejects_invalid_dimensions(bad):
    with pytest.raises(ValueError):
        robot_shape(length=bad)
    with pytest.raises(ValueError):
        robot_shape(width=bad)


def test_transform_robot_pose_translation():
    shape = robot_shape()
    np.testing.assert_allclose(transform_robot_pose(shape, 2.0, -1.0, 0.0), shape + np.array([[2.0], [-1.0]]))


def test_transform_robot_pose_rotation_quarter_turn():
    out = transform_robot_pose(np.array([[1.0], [0.0]]), 0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(out, [[0.0], [1.0]], atol=1e-15)


def test_transform_robot_pose_rejects_bad_input():
    with pytest.raises(ValueError):
        transform_robot_pose(np.zeros((3, 5)), 0, 0, 0)
    with pytest.raises(ValueError):
        transform_robot_pose(robot_shape(), np.nan, 0, 0)


def test_plot_planar_motion_writes_nonempty_file(tmp_path):
    path = tmp_path / "motion.png"
    plot_planar_motion(synthetic_run(), path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_animate_planar_motion_returns_animation():
    anim = animate_planar_motion(synthetic_run())
    assert isinstance(anim, FuncAnimation)
    plt.close(anim._fig)
    with warnings.catch_warnings():  # never rendered by design; silence only its deletion warning
        warnings.filterwarnings("ignore", message="Animation was deleted")
        del anim
        gc.collect()
