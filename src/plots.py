"""Figures for the KT3 scenarios. Each function takes ``runs``: {scenario id: dict with t, X, targets}."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
