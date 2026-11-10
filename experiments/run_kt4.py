"""KT4 experiment: working planar simulation of Baseline v1 -> figures, diagnostics, optional animation.

    python -m experiments.run_kt4 [--out-root results] [--save-animation]

Writes <out-root>/kt4_<UTC>/ as a new directory; existing results are never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.scenarios import SCENARIOS, SOLVER, run_scenario
from src.dataset import UNITS, save_targets
from src.plots import animate_planar_motion, plot_planar_motion, plot_residuals, plot_speeds, save_animation

ANIMATED = "S2_arc"


def build_runs() -> dict:
    """Run all KT3/KT4 scenarios; {scenario id: run dict}."""
    return {name: run_scenario(name, *rest) for name, *rest in SCENARIOS}


def build_diagnostics(runs: dict) -> dict:
    """Existing per-scenario diagnostics, unchanged and without pass/fail verdicts."""
    return {name: dict(run["diagnostics"]) for name, run in runs.items()}


def save_diagnostics(diagnostics: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def build_output_path(out_root: str | Path) -> Path:
    """Path of a new ``kt4_<UTC>`` directory under ``out_root``; not created here (``save_targets`` does it, never overwriting)."""
    out = Path(out_root) / f"kt4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if out.exists():
        raise FileExistsError(out)
    return out


def generate_figures(runs: dict, out: Path) -> None:
    for name, run in runs.items():
        plot_planar_motion(run, out / f"{name}_motion.png")
    plot_speeds(runs, out / "speeds.png")
    plot_residuals(runs, out / "residuals.png")


def generate_animation(runs: dict, out: Path, *, save: bool) -> None:
    """Build the S2_arc animation; write the GIF only if ``save``."""
    anim = animate_planar_motion(runs[ANIMATED])
    if save:
        save_animation(anim, out / f"{ANIMATED}_animation.gif")
    plt.close(anim._fig)
    with warnings.catch_warnings():  # an animation that is built but not rendered warns on deletion
        warnings.filterwarnings("ignore", message="Animation was deleted")
        del anim
        gc.collect()


def print_summary(runs: dict, out: Path) -> None:
    print("KT4 planar robot simulation\n" + "=" * 32 + f"\n\nModel: Baseline v1\nSolver: {SOLVER['method']}"
          f"\nScenarios: {len(runs)}")
    labels = (("max_abs_r_dense", "max |r|"), ("max_state_error_vs_closed_form", "closed-form error"),
              ("max_state_diff_vs_5state_model", "6D/5D difference"))
    for name, run in runs.items():
        print(f"\n{name}")
        for key, label in labels:
            if key in run["diagnostics"]:
                print(f"  {label:<19}{run['diagnostics'][key]:.3e}")
    print(f"\nResults: {out}")


def main(out_root: str | Path = "results", *, save_animation: bool = False) -> Path:
    out = build_output_path(out_root)
    runs = build_runs()
    command = "python -m experiments.run_kt4" + (" --save-animation" if save_animation else "")
    save_targets([r["target_set"] for r in runs.values()], out, metadata={
        "experiment": "KT4 working planar simulation",
        "model": "Baseline v1 (M2 practice model), X = (x, y, theta, vx, vy, omega), r = vx sin(theta) - vy cos(theta)",
        "command": command, "units": UNITS,
        "note": "synthetic simulation output; not a measurement of a physical robot"})
    save_diagnostics(build_diagnostics(runs), out / "diagnostics.json")
    generate_figures(runs, out)
    generate_animation(runs, out, save=save_animation)
    print_summary(runs, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--save-animation", action="store_true")
    args = ap.parse_args()
    main(args.out_root, save_animation=args.save_animation)
