import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "teacher_collect.py"
SPEC = importlib.util.spec_from_file_location("teacher_collect", MODULE_PATH)
teacher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher)


def test_smootherstep_boundaries_and_monotonicity():
    values = np.array([teacher.smootherstep(x) for x in np.linspace(-0.2, 1.2, 101)])
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all(np.diff(values) >= -1e-12)


def test_interpolation_endpoints():
    start = np.array([0.0, -0.15, 0.15, 0.0])
    target = np.array([0.35, -0.65, 0.50, -0.35])
    np.testing.assert_allclose(teacher.interpolate(start, target, 0.0), start)
    np.testing.assert_allclose(teacher.interpolate(start, target, 1.0), target)


def test_validate_arm_rejects_limit_violation():
    teacher.validate_arm([0.0, -0.15, 0.15, 0.0])
    with pytest.raises(ValueError):
        teacher.validate_arm([3.0, 0.0, 0.0, 0.0])
