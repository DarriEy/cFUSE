#!/usr/bin/env python3
"""
Unit tests for cfuse.calibration.parameter_manager.CFUSEParameterManager.

The parameter manager subclasses SYMFLUENCE's BaseParameterManager, so these
tests are skipped when symfluence is not installed. They exercise the cFUSE
parameter transformation logic (normalize/denormalize/validate/clip) which had
no test coverage.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("symfluence")

from cfuse.calibration.parameter_manager import CFUSEParameterManager  # noqa: E402


def _make_pm(params="S1_max,ku,b", extra_config=None):
    config = {
        "DOMAIN_NAME": "test",
        "EXPERIMENT_ID": "exp1",
        "CFUSE_PARAMS_TO_CALIBRATE": params,
    }
    if extra_config:
        config.update(extra_config)
    return CFUSEParameterManager(
        config, logging.getLogger("test_pm"), Path(tempfile.mkdtemp())
    )


class TestParameterSelection:
    def test_explicit_param_list(self):
        pm = _make_pm("S1_max,ku,b")
        assert pm.calibration_params == ["S1_max", "ku", "b"]

    def test_default_param_list(self):
        """'default' selects the 13-parameter calibration set."""
        pm = _make_pm("default")
        assert len(pm.calibration_params) == 13
        assert "S1_max" in pm.calibration_params
        assert "smooth_frac" in pm.calibration_params

    def test_whitespace_is_stripped(self):
        pm = _make_pm(" S1_max , ku ,b ")
        assert pm.calibration_params == ["S1_max", "ku", "b"]


class TestTransforms:
    def test_normalize_is_bounded(self):
        pm = _make_pm("S1_max,ku,b")
        # Values well outside bounds clip into [0, 1].
        norm = pm.normalize({"S1_max": 1e9, "ku": -1e9, "b": 1.5})
        assert norm.min() >= 0.0 and norm.max() <= 1.0

    def test_midpoint_normalizes_to_half(self):
        pm = _make_pm("S1_max")            # bounds (50, 5000) -> midpoint 2525
        norm = pm.normalize({"S1_max": 2525.0})
        assert norm[0] == pytest.approx(0.5, abs=1e-3)

    def test_denormalize_roundtrip(self):
        pm = _make_pm("S1_max,ku,b")
        original = {"S1_max": 200.0, "ku": 10.0, "b": 1.5}
        recovered = pm.denormalize(pm.normalize(original))
        for k, v in original.items():
            assert recovered[k] == pytest.approx(v, rel=1e-4)

    def test_dict_array_roundtrip(self):
        pm = _make_pm("S1_max,ku,b")
        arr = pm.dict_to_array({"S1_max": 200.0, "ku": 10.0, "b": 1.5})
        assert arr.tolist() == [200.0, 10.0, 1.5]
        assert pm.array_to_dict(arr) == {"S1_max": 200.0, "ku": 10.0, "b": 1.5}

    def test_bounds_array_shapes(self):
        pm = _make_pm("S1_max,ku,b")
        lower, upper = pm.get_bounds_array()
        assert lower.shape == upper.shape == (3,)
        assert (lower < upper).all()


class TestValidation:
    def test_validate_accepts_in_bounds(self):
        pm = _make_pm("S1_max,ku,b")
        ok, violations = pm.validate({"S1_max": 200.0, "ku": 10.0, "b": 1.5})
        assert ok and violations == []

    def test_validate_flags_out_of_bounds(self):
        pm = _make_pm("S1_max,ku,b")
        ok, violations = pm.validate({"S1_max": 1e9, "ku": 10.0, "b": 1.5})
        assert not ok and len(violations) == 1

    def test_clip_to_bounds(self):
        pm = _make_pm("S1_max")
        clipped = pm.clip_to_bounds({"S1_max": 1e9})
        low, high = pm.get_bounds("S1_max")
        assert clipped["S1_max"] == pytest.approx(high)

    def test_get_complete_params_fills_defaults(self):
        pm = _make_pm("S1_max,ku,b")
        complete = pm.get_complete_params({"S1_max": 123.0})
        assert complete["S1_max"] == 123.0
        # Untouched params fall back to defaults for the full parameter set.
        assert len(complete) >= len(pm.calibration_params)
        assert "ku" in complete


class TestCustomBounds:
    def test_config_override_bounds(self):
        pm = _make_pm(
            "S1_max,ku,b",
            extra_config={"CFUSE_PARAM_BOUNDS": {"S1_max": [100.0, 200.0]}},
        )
        assert pm.get_bounds("S1_max") == (100.0, 200.0)
