"""KT3 experiment: scenarios -> trajectories -> target states T_i = X(t_i) -> diagnostics -> figures.

    python -m experiments.run_kt3 [--out-root results]

Writes <out-root>/kt3_<UTC>/ as a new directory; existing results are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from experiments.scenarios import SCENARIOS, run_scenario
from src.dataset import UNITS, code_version, save_targets
from src.plots import plot_paths, plot_residuals, plot_speeds


def main(out_root="results") -> Path:
    runs = {name: run_scenario(name, *rest) for name, *rest in SCENARIOS}
    out = Path(out_root) / f"kt3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    save_targets([r["target_set"] for r in runs.values()], out, metadata={
        "model": "Baseline v1 (M2 practice model), X = (x, y, theta, vx, vy, omega), r = vx sin(theta) - vy cos(theta)",
        "command": "python -m experiments.run_kt3", "units": UNITS,
        "note": "synthetic simulation output; not a measurement of a physical robot"})
    plot_paths(runs, out / "paths.png")
    plot_speeds(runs, out / "speeds.png")
    plot_residuals(runs, out / "residuals.png")
    for name, r in runs.items():
        print(name, r["target_set"].metadata["diagnostics"])
    print(f"written to {out} (commit {code_version()['commit']})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default="results")
    main(ap.parse_args().out_root)
