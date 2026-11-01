import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "convert_pick_place_dataset.py"
SPEC = importlib.util.spec_from_file_location("convert_pick_place_dataset", MODULE_PATH)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def test_sample_and_hold_uses_latest_previous_value():
    times = np.array([10, 20, 40])
    values = np.array([[1.0], [2.0], [4.0]])
    targets = np.array([10, 19, 20, 39, 40])
    actual = converter.sample_hold(times, values, targets).flatten()
    np.testing.assert_array_equal(actual, [1.0, 1.0, 2.0, 2.0, 4.0])


def test_resample_keeps_five_dimensions_and_no_padding():
    times = np.array([0, 50_000_000, 100_000_000, 150_000_000])
    states = np.arange(20, dtype=np.float32).reshape(4, 5)
    arm = np.arange(16, dtype=np.float32).reshape(4, 4)
    gripper = np.arange(4, dtype=np.float32)
    out_states, out_actions, out_times = converter.resample(
        {
            "state": list(zip(times, states)),
            "action": list(zip(times, np.c_[arm, gripper])),
            "stage": [(0, "approach"), (150_000_000, "retreat")],
        },
        fps=20,
    )
    assert out_states.shape == (4, 5)
    assert out_actions.shape == (4, 5)
    assert out_times.shape == (4,)
