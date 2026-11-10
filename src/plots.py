"""Figures for the KT3/KT4 scenarios. ``runs``: {scenario id: dict with t, X, targets}; a single ``run`` also has name."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from src.metrics import residual_drift  # noqa: E402
from src.model import body_speed  # noqa: E402


def plot_paths(runs: dict, path) -> None:
    """Planar paths y(x) with the target positions T_i marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, (name, run) in enumerate(runs.items()):
        line, = ax.plot(run["X"][0], run["X"][1], label=name)
        ax.plot(run["targets"][0], run["targets"][1], "o", color=line.get_color())
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Траектории; маркеры: положения $(x_i, y_i)$ целевых состояний")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speeds(runs: dict, path) -> None:
    """Body-axis speed v(t) and yaw rate omega(t) for each scenario."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for name, run in runs.items():
        a1.plot(run["t"], body_speed(run["X"]), label=name)
        a2.plot(run["t"], run["X"][5], label=name)
    a1.set_ylabel("v(t), м/с")
    a2.set_ylabel("ω(t), рад/с")
    a2.set_xlabel("t, с")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend()
    a1.set_title("Скорость вдоль продольной оси и угловая скорость")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(runs: dict, path) -> None:
    """Constraint drift |r(t) - r(0)| [m/s] on a log axis (integration error, not model error)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, run in runs.items():
        ax.semilogy(run["t"], residual_drift(run["X"]) + 1e-18, label=name)
    ax.set_xlabel("t, с")
    ax.set_ylabel("|r(t) − r(0)|, м/с")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Дрейф невязки неголономной связи")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robot_shape(length: float = 0.40, width: float = 0.24) -> np.ndarray:
    """Closed body-frame rectangle, shape (2, 5), front edge at +length/2. Graphical size only."""
    if not (np.isfinite(length) and np.isfinite(width) and length > 0 and width > 0):
        raise ValueError("length and width must be finite and positive")
    L, W = length / 2, width / 2
    return np.array([[L, L, -L, -L, L], [W, -W, -W, W, W]])


def transform_robot_pose(shape: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    """World-frame points R(theta) @ shape + (x, y); ``shape`` is (2, N)."""
    shape = np.asarray(shape, dtype=float)
    if shape.ndim != 2 or shape.shape[0] != 2 or not np.all(np.isfinite(shape)):
        raise ValueError("shape must be a finite array of shape (2, N)")
    if not all(np.isfinite(v) for v in (x, y, theta)):
        raise ValueError("x, y, theta must be finite")
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]]) @ shape + np.array([[x], [y]])


def _draw_pose(ax, shape, x, y, theta, **kw):
    """Draw the body outline and a heading segment (centre -> front edge); return the two artists."""
    body = transform_robot_pose(shape, x, y, theta)
    head = transform_robot_pose(np.array([[0.0, shape[0, 0]], [0.0, 0.0]]), x, y, theta)
    return ax.plot(*body, **kw)[0], ax.plot(*head, **kw)[0]


def plot_planar_motion(run: dict, path, *, pose_indices=None, robot_length: float = 0.40,
                       robot_width: float = 0.24) -> None:
    """Planar path of one run with robot poses (x, y, theta); markers T_i are samples of the same trajectory."""
    t, X = np.asarray(run["t"]), np.asarray(run["X"])
    if pose_indices is None:
        pose_indices = np.linspace(0, t.size - 1, 5, dtype=int)
    shape = robot_shape(robot_length, robot_width)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(X[0], X[1], color="C0", lw=1.2, label="Траектория")
    ax.plot(X[0, 0], X[1, 0], "s", color="C2", ms=7, label="Начальное положение")
    ax.plot(X[0, -1], X[1, -1], "*", color="C3", ms=11, label="Конечное положение")
    ax.plot(run["targets"][0], run["targets"][1], "o", mfc="none", color="C1", label="Проекции состояний $T_i$")
    for k, i in enumerate(pose_indices):
        _draw_pose(ax, shape, X[0, i], X[1, i], X[2, i], color="k", lw=1.2)
        ax.annotate(f"{t[i]:g} с", (X[0, i], X[1, i]), textcoords="offset points", xytext=(12, 10), fontsize=7, ha="left")
    ax.margins(0.08)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Движение мобильного робота в плоскости: {run['name']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def animate_planar_motion(run: dict, *, interval_ms: int = 50, trail: bool = True, robot_length: float = 0.40,
                          robot_width: float = 0.24) -> FuncAnimation:
    """Animation of the pose (x, y, theta) along the simulated trajectory; the data are not altered."""
    t, X = np.asarray(run["t"]), np.asarray(run["X"])
    shape = robot_shape(robot_length, robot_width)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(X[0], X[1], color="0.8", lw=1, label="Траектория")
    trail_line, = ax.plot([], [], color="C0", lw=1.5)
    body, head = _draw_pose(ax, shape, X[0, 0], X[1, 0], X[2, 0], color="k", lw=1.5)
    label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.05)
    ax.grid(alpha=0.3)
    ax.set_title(f"Движение мобильного робота в плоскости: {run['name']}")
    frames = np.arange(0, t.size, 2)

    def update(i):
        pts = transform_robot_pose(shape, X[0, i], X[1, i], X[2, i])
        nose = transform_robot_pose(np.array([[0.0, shape[0, 0]], [0.0, 0.0]]), X[0, i], X[1, i], X[2, i])
        body.set_data(*pts)
        head.set_data(*nose)
        if trail:
            trail_line.set_data(X[0, :i + 1], X[1, :i + 1])
        label.set_text(f"t = {t[i]:.2f} с")
        return body, head, trail_line, label

    return FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)


def save_animation(animation, path, *, fps: int = 20) -> None:
    """Save as GIF through Matplotlib's Pillow writer."""
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(animation._fig)
