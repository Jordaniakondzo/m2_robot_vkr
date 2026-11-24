"""KT5 experiment: frozen 22-run parameter campaign on Baseline v1 -> trajectories, metrics, metadata.

    python -m experiments.run_kt5 [--out-root results]

The matrix follows private/KT5_EXPERIMENT_PLAN.md and is not configurable. Writes <out-root>/kt5_<UTC>/
as a new directory; existing results are never overwritten. Collection only: no analysis or verdicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import simpson

from src.dataset import UNITS, code_version
from src.metrics import constraint_residual
from src.model import Params, body_speed, constant_control, rhs
from src.solver import integrate

T0, TF, N_POINTS = 0.0, 10.0, 201
T = np.linspace(T0, TF, N_POINTS)
SOLVER = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}

FACTORS = ("m", "J", "v0", "omega0", "F", "tau")
FIXED = {"x0": 0.0, "y0": 0.0, "theta0": 0.0}
REF = {"m": 10.0, "J": 2.0, "v0": 1.0, "omega0": 0.3, "F": 2.0, "tau": 0.2}
COMBINED_F_TAU = {(0.0, 0.0), (4.0, 0.0), (0.0, 0.4), (4.0, 0.4)}
EXPECTED_COUNTS = {"reference": 1, "ofat": 17, "combined": 4}

# run_id, group, factor, m, J, v0, omega0, F, tau -- frozen order, copied from the plan.
_MATRIX = [
    ("REF", "reference", "reference", 10, 2, 1.0, 0.3, 2, 0.2),
    ("V0_0p5", "ofat", "v0", 10, 2, 0.5, 0.3, 2, 0.2),
    ("V0_1p5", "ofat", "v0", 10, 2, 1.5, 0.3, 2, 0.2),
    ("V0_2p0", "ofat", "v0", 10, 2, 2.0, 0.3, 2, 0.2),
    ("W0_m0p6", "ofat", "omega0", 10, 2, 1.0, -0.6, 2, 0.2),
    ("W0_m0p3", "ofat", "omega0", 10, 2, 1.0, -0.3, 2, 0.2),
    ("W0_0p0", "ofat", "omega0", 10, 2, 1.0, 0.0, 2, 0.2),
    ("W0_0p6", "ofat", "omega0", 10, 2, 1.0, 0.6, 2, 0.2),
    ("M_5", "ofat", "m", 5, 2, 1.0, 0.3, 2, 0.2),
    ("M_20", "ofat", "m", 20, 2, 1.0, 0.3, 2, 0.2),
    ("J_1", "ofat", "J", 10, 1, 1.0, 0.3, 2, 0.2),
    ("J_4", "ofat", "J", 10, 4, 1.0, 0.3, 2, 0.2),
    ("F_0", "ofat", "F", 10, 2, 1.0, 0.3, 0, 0.2),
    ("F_1", "ofat", "F", 10, 2, 1.0, 0.3, 1, 0.2),
    ("F_4", "ofat", "F", 10, 2, 1.0, 0.3, 4, 0.2),
    ("TAU_m0p2", "ofat", "tau", 10, 2, 1.0, 0.3, 2, -0.2),
    ("TAU_0p0", "ofat", "tau", 10, 2, 1.0, 0.3, 2, 0.0),
    ("TAU_0p4", "ofat", "tau", 10, 2, 1.0, 0.3, 2, 0.4),
    ("C_F0_T0", "combined", "F_tau", 10, 2, 1.0, 0.3, 0, 0.0),
    ("C_F4_T0", "combined", "F_tau", 10, 2, 1.0, 0.3, 4, 0.0),
    ("C_F0_T0p4", "combined", "F_tau", 10, 2, 1.0, 0.3, 0, 0.4),
    ("C_F4_T0p4", "combined", "F_tau", 10, 2, 1.0, 0.3, 4, 0.4),
]

EXPERIMENT_COLUMNS = ["run_id", "group", "factor", "factor_value", "m", "J", "x0", "y0", "theta0", "v0", "omega0",
                      "F", "tau", "t0", "tf", "n_points", "solver", "rtol", "atol"]
TRAJECTORY_COLUMNS = ["run_id", "t", "x", "y", "theta", "vx", "vy", "omega", "v", "r"]
METRIC_COLUMNS = ["run_id", "x_final", "y_final", "theta_final", "v_final", "omega_final", "distance_final",
                  "path_length", "delta_theta", "max_abs_r", "max_drift", "expected_v_final",
                  "expected_omega_final", "expected_theta_final", "error_v", "error_omega", "error_theta"]
SUMMARY_METRICS = ("max_abs_r", "max_drift", "error_v", "error_omega", "error_theta")


def build_experiment_matrix() -> list[dict]:
    """The 22 frozen run specifications, in plan order. No integration, no I/O."""
    specs = []
    for run_id, group, factor, *values in _MATRIX:
        spec = {"run_id": run_id, "group": group, "factor": factor, "factor_value": None, **FIXED}
        spec.update({k: float(v) for k, v in zip(("m", "J", "v0", "omega0", "F", "tau"), values)})
        if group == "ofat":
            spec["factor_value"] = spec[factor]
        specs.append(spec)
    return specs


def validate_experiment_matrix(experiments: list[dict]) -> None:
    """Raise ValueError if the matrix deviates from the frozen design (counts, OFAT/combined structure, values)."""
    if len(experiments) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"expected 22 runs, got {len(experiments)}")
    ids = [e["run_id"] for e in experiments]
    if len(set(ids)) != len(ids):
        raise ValueError("run_id values are not unique")
    configs = [tuple(e[k] for k in (*FACTORS, *FIXED)) for e in experiments]
    if len(set(configs)) != len(configs):
        raise ValueError("physical configurations are not unique")
    for e in experiments:
        for k in (*FACTORS, *FIXED):
            if not (isinstance(e[k], float) and math.isfinite(e[k])):
                raise ValueError(f"{e['run_id']}: {k} must be a finite float, got {e[k]!r}")
        if e["m"] <= 0 or e["J"] <= 0:
            raise ValueError(f"{e['run_id']}: m and J must be > 0")
    counts = {g: sum(e["group"] == g for e in experiments) for g in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"group counts {counts} != {EXPECTED_COUNTS}")
    for e in experiments:
        diff = [k for k in FACTORS if e[k] != REF[k]]
        if e["group"] == "reference":
            ok = not diff and e["run_id"] == "REF" and e["factor"] == "reference" and e["factor_value"] is None
        elif e["group"] == "ofat":
            ok = diff == [e["factor"]] and e["factor"] in FACTORS and e["factor_value"] == e[e["factor"]]
        else:
            ok = (e["factor"] == "F_tau" and e["factor_value"] is None and set(diff) <= {"F", "tau"}
                  and (e["F"], e["tau"]) in COMBINED_F_TAU)
        if not ok:
            raise ValueError(f"{e['run_id']}: inconsistent with group {e['group']!r} / factor {e['factor']!r}")
    if {(e["F"], e["tau"]) for e in experiments if e["group"] == "combined"} != COMBINED_F_TAU:
        raise ValueError("combined runs do not match the four planned F/tau cases")


def initial_state(spec: dict) -> np.ndarray:
    """X0 = (x0, y0, theta0, v0 cos(theta0), v0 sin(theta0), omega0), so that r(X0) = 0."""
    th, v0 = spec["theta0"], spec["v0"]
    return np.array([spec["x0"], spec["y0"], th, v0 * math.cos(th), v0 * math.sin(th), spec["omega0"]])


def run_experiment(spec: dict, *, t: np.ndarray = T, solver: dict = SOLVER) -> dict:
    """Integrate one run with Baseline v1; returns {spec, t, X (6, N), v (N,), r (N,)} in memory."""
    params = Params(spec["m"], spec["J"])
    control = constant_control(spec["F"], spec["tau"])
    X = integrate(rhs, initial_state(spec), t, params, control, **solver)
    return {"spec": spec, "t": t, "X": X, "v": body_speed(X), "r": constraint_residual(X)}


def expected_final_state(spec: dict, tf: float = TF) -> dict:
    """Closed-form v, omega, theta at tf for constant F and tau (t0 = 0)."""
    return {"expected_v_final": spec["v0"] + spec["F"] / spec["m"] * tf,
            "expected_omega_final": spec["omega0"] + spec["tau"] / spec["J"] * tf,
            "expected_theta_final": spec["theta0"] + spec["omega0"] * tf + spec["tau"] / (2 * spec["J"]) * tf**2}


def compute_metrics(run: dict) -> dict:
    """Final-state, path, rotation, constraint and closed-form-error metrics for one run (no verdicts)."""
    spec, t, X, v, r = run["spec"], run["t"], run["X"], run["v"], run["r"]
    exp = expected_final_state(spec, float(t[-1] - t[0]))
    x_f, y_f, th_f, om_f, v_f = X[0, -1], X[1, -1], X[2, -1], X[5, -1], v[-1]
    return {"run_id": spec["run_id"], "x_final": x_f, "y_final": y_f, "theta_final": th_f, "v_final": v_f,
            "omega_final": om_f, "distance_final": math.hypot(x_f - spec["x0"], y_f - spec["y0"]),
            "path_length": float(simpson(np.hypot(X[3], X[4]), x=t)), "delta_theta": th_f - spec["theta0"],
            "max_abs_r": float(np.max(np.abs(r))), "max_drift": float(np.max(np.abs(r - r[0]))), **exp,
            "error_v": abs(v_f - exp["expected_v_final"]), "error_omega": abs(om_f - exp["expected_omega_final"]),
            "error_theta": abs(th_f - exp["expected_theta_final"])}


def build_runs() -> dict[str, dict]:
    """Matrix -> validation -> 22 integrations; {run_id: run}, in matrix order."""
    experiments = build_experiment_matrix()
    validate_experiment_matrix(experiments)
    return {e["run_id"]: run_experiment(e) for e in experiments}


def build_experiment_rows(runs: dict[str, dict]) -> list[dict]:
    return [{**{k: run["spec"][k] for k in EXPERIMENT_COLUMNS[:13]}, "t0": float(run["t"][0]),
             "tf": float(run["t"][-1]), "n_points": len(run["t"]), "solver": SOLVER["method"],
             "rtol": SOLVER["rtol"], "atol": SOLVER["atol"]} for run in runs.values()]


def build_trajectory_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id, run in runs.items():
        X = run["X"]
        for i, ti in enumerate(run["t"]):
            rows.append({"run_id": run_id, "t": float(ti), "x": X[0, i], "y": X[1, i], "theta": X[2, i],
                         "vx": X[3, i], "vy": X[4, i], "omega": X[5, i], "v": run["v"][i], "r": run["r"][i]})
    return rows


def build_metric_rows(runs: dict[str, dict]) -> list[dict]:
    return [compute_metrics(run) for run in runs.values()]


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: "" if row[c] is None else float(row[c]) if isinstance(row[c], np.floating) else row[c]
                             for c in columns})


def build_metadata(runs: dict[str, dict], command: str, code: dict) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "code": code,
        "software": {"python_version": platform.python_version(), "numpy_version": np.__version__,
                     "scipy_version": scipy.__version__},
        "experiment": "KT5 parameter sensitivity and data collection",
        "model": "Baseline v1 (M2 practice model), X = (x, y, theta, vx, vy, omega), r = vx sin(theta) - vy cos(theta)",
        "design": "OFAT with selected combined F/tau cases",
        "solver": dict(SOLVER),
        "time_grid": {"t0": T0, "tf": TF, "n_points": N_POINTS, "spacing": "uniform, endpoints included"},
        "run_count": len(runs), "groups": dict(EXPECTED_COUNTS),
        "units": {**UNITS, "m": "kg", "J": "kg m^2", "v0": "m/s", "omega0": "rad/s", "F": "N", "tau": "N m",
                  "v": "m/s", "r": "m/s", "path_length": "m", "distance_final": "m", "delta_theta": "rad"},
        "path_length_quadrature": "Simpson on the recorded grid, integrand sqrt(vx^2 + vy^2)",
        "experiment_matrix": [run["spec"] for run in runs.values()],
        "files": {"experiment_runs": "experiment_runs.csv", "trajectories": "trajectories.csv",
                  "metrics": "metrics.csv", "metadata": "metadata.json"},
        "command": command,
        "note": "synthetic simulation output; not a measurement of a physical robot"}


def build_output_path(out_root: str | Path) -> Path:
    """Path of a new ``kt5_<UTC>`` directory under ``out_root``; not created here."""
    out = Path(out_root) / f"kt5_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if out.exists():
        raise FileExistsError(out)
    return out


def save_campaign(runs: dict[str, dict], out: Path, command: str = "python -m experiments.run_kt5") -> Path:
    """Write the four campaign files into the new directory ``out``."""
    code = code_version()  # before mkdir, so the new directory does not make the tree dirty
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "experiment_runs.csv", EXPERIMENT_COLUMNS, build_experiment_rows(runs))
    _write_csv(out / "trajectories.csv", TRAJECTORY_COLUMNS, build_trajectory_rows(runs))
    _write_csv(out / "metrics.csv", METRIC_COLUMNS, build_metric_rows(runs))
    meta = build_metadata(runs, command, code)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False),
                                       encoding="utf-8")
    return out


def print_summary(runs: dict[str, dict], out: Path) -> None:
    metrics = build_metric_rows(runs)
    n = len(next(iter(runs.values()))["t"])
    print("KT5 parameter sensitivity campaign\n" + "=" * 34)
    print(f"Runs: {len(runs)}\nTime points per run: {n}\nTrajectory rows: {len(runs) * n}\n")
    for group in EXPECTED_COUNTS:
        print(f"{group}: {sum(r['spec']['group'] == group for r in runs.values())}")
    print("\nMaximum over runs:")
    for key in SUMMARY_METRICS:
        print(f"  {key:<12}{max(m[key] for m in metrics):.3e}")
    print(f"\nResults: {out}")


def main(out_root: str | Path = "results") -> Path:
    out = build_output_path(out_root)
    runs = build_runs()
    save_campaign(runs, out)
    print_summary(runs, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default="results")
    main(ap.parse_args().out_root)
