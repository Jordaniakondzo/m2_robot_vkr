"""Dataset schema, save/load round trip and an end-to-end run of the KT3 experiment."""

import json
import platform

import numpy as np
import pytest
import scipy

from experiments import run_kt3
from src.dataset import COLUMNS, TargetSet, load_targets, sample_targets, save_targets
from src.metrics import constraint_residual
from src.model import constant_control


def _toy():
    t = np.linspace(0.0, 4.0, 9)
    X = np.vstack([t, 2 * t, 0.1 * t, np.cos(0.1 * t), np.sin(0.1 * t), 0.1 * np.ones_like(t)])
    return sample_targets("toy", t, X, [1.0, 2.0, 4.0], constant_control(2.0, 0.2), {"note": "toy"})


def test_sample_targets_shapes_and_inputs():
    ts = _toy()
    assert ts.t.shape == (3,) and ts.X.shape == (6, 3) and ts.inputs.shape == (2, 3)
    np.testing.assert_array_equal(ts.inputs, [[2.0] * 3, [0.2] * 3])
    with pytest.raises(ValueError):
        sample_targets("bad", np.linspace(0, 4, 9), np.zeros((6, 9)), [1.1], constant_control(0, 0))


def test_roundtrip_is_bit_exact_and_never_overwrites(tmp_path):
    ts = _toy()
    out = save_targets([ts], tmp_path / "ds", {"purpose": "test"})
    (back,) = load_targets(out)
    np.testing.assert_array_equal(back.t, ts.t)
    np.testing.assert_array_equal(back.X, ts.X)
    np.testing.assert_array_equal(back.inputs, ts.inputs)
    assert back.metadata == ts.metadata
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["software"] == {"python_version": platform.python_version(), "numpy_version": np.__version__,
                                 "scipy_version": scipy.__version__}
    assert set(meta["code"]) == {"commit", "dirty"}
    assert tuple(next(open(out / "targets.csv")).strip().split(",")) == COLUMNS
    with pytest.raises(FileExistsError):
        save_targets([ts], out)


def test_save_rejects_duplicate_ids_and_bad_shapes(tmp_path):
    ts = _toy()
    with pytest.raises(ValueError):
        save_targets([ts, ts], tmp_path / "a")
    with pytest.raises(ValueError):
        save_targets([TargetSet("x", ts.t, ts.X[:5], ts.inputs)], tmp_path / "b")


def test_load_rejects_schema_violation(tmp_path):
    out = save_targets([_toy()], tmp_path / "ds")
    p = out / "targets.csv"
    p.write_text(p.read_text().replace("trajectory_id,t,x", "trajectory_id,t,X", 1))
    with pytest.raises(ValueError):
        load_targets(out)


def test_kt3_end_to_end(tmp_path):
    out = run_kt3.main(tmp_path)
    sets = {ts.trajectory_id: ts for ts in load_targets(out)}
    assert set(sets) == {"S1_line", "S2_arc", "S3_const_F_tau"}
    for name, ts in sets.items():
        assert ts.t.tolist() == pytest.approx([2, 4, 6, 8, 10]) and ts.X.shape == (6, 5)
        diag = ts.metadata["diagnostics"]
        assert np.max(np.abs(constraint_residual(ts.X))) < 1e-8
        assert diag["max_state_diff_vs_5state_model"] < 1e-7
        if name != "S3_const_F_tau":
            assert diag["max_state_error_vs_closed_form"] < 1e-7
    assert {"paths.png", "speeds.png", "residuals.png"} <= {p.name for p in out.iterdir()}
    json.loads((out / "metadata.json").read_text())


def _corrupt(tmp_path, row, col, value):
    """Save the toy dataset, then overwrite one CSV field (row index from 0, column name)."""
    out = save_targets([_toy()], tmp_path / "ds")
    lines = (out / "targets.csv").read_text().splitlines()
    fields = lines[1 + row].split(",")
    fields[COLUMNS.index(col)] = value
    lines[1 + row] = ",".join(fields)
    (out / "targets.csv").write_text("\n".join(lines) + "\n")
    return out


@pytest.mark.parametrize("row,col,value", [
    (0, "t", "nan"),         # non-finite time
    (1, "t", "0.5"),         # time not increasing (t = 1, 0.5, 4)
    (2, "t", "2.0"),         # time equal to the previous one
    (1, "x", "inf"),         # non-finite state
    (1, "omega", "nan"),     # non-finite state
    (1, "F", "inf"),         # non-finite command
    (1, "tau", "nan"),       # non-finite command
    (1, "r", "0.5"),         # stored r differs from r recomputed from the state
])
def test_load_rejects_corrupted_values(tmp_path, row, col, value):
    with pytest.raises(ValueError):
        load_targets(_corrupt(tmp_path, row, col, value))


def test_load_accepts_uncorrupted_values(tmp_path):
    """Control for the corruption helper: rewriting a field with its own value must load."""
    out = _corrupt(tmp_path, 1, "tau", "0.2")
    (ts,) = load_targets(out)
    assert ts.inputs[1, 1] == 0.2
