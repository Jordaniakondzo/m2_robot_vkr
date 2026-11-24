"""KT5 campaign: matrix fidelity, initial state, metrics, collected data and saved files."""

import csv
import json
import math

import numpy as np
import pytest

from experiments import run_kt5 as k5
from src.metrics import constraint_residual


@pytest.fixture(scope="module")
def matrix():
    return k5.build_experiment_matrix()


@pytest.fixture(scope="module")
def ref_run(matrix):
    return k5.run_experiment(matrix[0])


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    runs = k5.build_runs()
    return runs, k5.save_campaign(runs, tmp_path_factory.mktemp("kt5") / "kt5_test")


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_matrix_size_ids_configs_and_groups(matrix):
    k5.validate_experiment_matrix(matrix)
    assert len(matrix) == 22
    assert len({e["run_id"] for e in matrix}) == 22
    assert len({tuple(e[k] for k in k5.FACTORS) for e in matrix}) == 22
    assert [sum(e["group"] == g for e in matrix) for g in ("reference", "ofat", "combined")] == [1, 17, 4]


def test_ref_parameters(matrix):
    ref = matrix[0]
    assert ref["run_id"] == "REF"
    assert (ref["m"], ref["J"], ref["v0"], ref["omega0"], ref["F"], ref["tau"]) == (10.0, 2.0, 1.0, 0.3, 2.0, 0.2)
    assert (ref["x0"], ref["y0"], ref["theta0"]) == (0.0, 0.0, 0.0)


def test_ofat_runs_change_exactly_one_factor(matrix):
    for e in matrix:
        if e["group"] == "ofat":
            assert [k for k in k5.FACTORS if e[k] != k5.REF[k]] == [e["factor"]]
            assert e["factor_value"] == e[e["factor"]]


def test_combined_cases_are_the_planned_four(matrix):
    combined = {(e["F"], e["tau"]) for e in matrix if e["group"] == "combined"}
    assert combined == {(0.0, 0.0), (4.0, 0.0), (0.0, 0.4), (4.0, 0.4)}


def test_validation_rejects_broken_matrices(matrix):
    with pytest.raises(ValueError):
        k5.validate_experiment_matrix(matrix[:-1])
    broken = [dict(e) for e in matrix]
    broken[1]["m"] = 5.0  # OFAT run now changes two factors
    with pytest.raises(ValueError):
        k5.validate_experiment_matrix(broken)


def test_initial_state_with_nonzero_theta():
    spec = {"x0": 1.0, "y0": -2.0, "theta0": 0.7, "v0": 2.0, "omega0": -0.4}
    X0 = k5.initial_state(spec)
    assert np.allclose(X0, [1.0, -2.0, 0.7, 2.0 * math.cos(0.7), 2.0 * math.sin(0.7), -0.4])
    assert abs(constraint_residual(X0)) < 1e-15


def test_initial_state_satisfies_constraint(matrix):
    assert all(abs(constraint_residual(k5.initial_state(e))) < 1e-15 for e in matrix)


def test_expected_final_state_ref(matrix):
    exp = k5.expected_final_state(matrix[0])
    assert exp["expected_v_final"] == pytest.approx(3.0)
    assert exp["expected_omega_final"] == pytest.approx(1.3)
    assert exp["expected_theta_final"] == pytest.approx(8.0)


def test_compute_metrics_on_artificial_run():
    spec = {"run_id": "ART", "x0": 0.0, "y0": 0.0, "theta0": 0.0, "v0": 2.0, "omega0": 0.0,
            "m": 1.0, "J": 1.0, "F": 0.0, "tau": 0.0}
    t = np.linspace(0.0, 10.0, 201)
    X = np.zeros((6, 201))
    X[0], X[3] = 2.0 * t, 2.0                      # uniform motion along x
    X[2] = 7.0                                     # unwrapped angle, theta0 = 0
    run = {"spec": spec, "t": t, "X": X, "v": X[3] * np.cos(X[2]) + X[4] * np.sin(X[2]),
           "r": constraint_residual(X)}
    m = k5.compute_metrics(run)
    assert m["x_final"] == pytest.approx(20.0) and m["distance_final"] == pytest.approx(20.0)
    assert m["path_length"] == pytest.approx(20.0)
    assert m["delta_theta"] == 7.0                 # not wrapped
    assert m["v_final"] == pytest.approx(2.0 * math.cos(7.0))
    assert m["expected_v_final"] == 2.0 and m["expected_theta_final"] == 0.0
    assert m["error_theta"] == 7.0 and m["error_omega"] == 0.0
    assert m["max_abs_r"] == pytest.approx(2.0 * abs(math.sin(7.0)))
    assert m["max_drift"] == pytest.approx(0.0, abs=1e-15)


def test_ref_run_shapes_finite_and_closed_form(ref_run):
    assert ref_run["X"].shape == (6, 201) and ref_run["v"].shape == (201,) and ref_run["r"].shape == (201,)
    assert np.all(np.isfinite(ref_run["X"])) and np.all(np.diff(ref_run["t"]) > 0)
    m = k5.compute_metrics(ref_run)
    assert m["v_final"] == pytest.approx(3.0, abs=1e-8)
    assert m["theta_final"] == pytest.approx(8.0, abs=1e-8)
    assert m["delta_theta"] == pytest.approx(8.0, abs=1e-8)


def test_campaign_sizes_and_files(campaign):
    runs, out = campaign
    assert len(runs) == 22
    assert sorted(p.name for p in out.iterdir()) == ["experiment_runs.csv", "metadata.json", "metrics.csv",
                                                     "trajectories.csv"]
    exp, traj, met = (_read(out / f) for f in ("experiment_runs.csv", "trajectories.csv", "metrics.csv"))
    assert (len(exp), len(traj), len(met)) == (22, 4422, 22)
    assert list(traj[0]) == k5.TRAJECTORY_COLUMNS and list(exp[0]) == k5.EXPERIMENT_COLUMNS
    assert [r["run_id"] for r in exp] == list(runs) == [r["run_id"] for r in met]


def test_trajectory_roundtrip_and_ordering(campaign):
    runs, out = campaign
    traj = _read(out / "trajectories.csv")
    assert [r["run_id"] for r in traj[::201]] == list(runs)
    ref = [r for r in traj if r["run_id"] == "REF"]
    assert np.array_equal([float(r["x"]) for r in ref], runs["REF"]["X"][0])  # lossless text round trip
    assert np.all(np.diff([float(r["t"]) for r in ref]) > 0)


def test_metadata_valid_json_and_no_nonfinite(campaign):
    _, out = campaign
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(c))
    for key in ("created_utc", "code", "software", "experiment", "model", "design", "solver", "time_grid",
                "run_count", "groups", "units", "files", "command", "note"):
        assert key in meta
    assert meta["run_count"] == 22 and len(meta["experiment_matrix"]) == 22
    assert meta["groups"] == {"reference": 1, "ofat": 17, "combined": 4}
    for name in ("experiment_runs.csv", "trajectories.csv", "metrics.csv"):
        for row in _read(out / name):
            for k, v in row.items():
                if k not in ("run_id", "group", "factor", "solver", "factor_value") or (k == "factor_value" and v):
                    assert math.isfinite(float(v)), (name, k, v)


def test_output_path_is_new_and_not_overwritten(tmp_path):
    out = k5.build_output_path(tmp_path)
    assert out.parent == tmp_path and out.name.startswith("kt5_") and not out.exists()
    out.mkdir()
    with pytest.raises(FileExistsError):
        k5.save_campaign({}, out)
