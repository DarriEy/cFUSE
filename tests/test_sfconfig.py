#!/usr/bin/env python3
"""
Tests for the cFUSE SYMFLUENCE config schema (cfuse.sfconfig.CFUSEConfig).

CFUSEConfig is a pydantic model that subclasses a SYMFLUENCE base adapter, so
these tests are skipped when symfluence is not installed.
"""

import pytest

pytest.importorskip("symfluence")

from pydantic import ValidationError  # noqa: E402

from cfuse.sfconfig import CFUSEConfig, DEFAULT_CALIBRATION_PARAMS  # noqa: E402


class TestDefaults:
    def test_defaults(self):
        cfg = CFUSEConfig()
        assert cfg.model_structure == "prms"
        assert cfg.enable_snow is True
        assert cfg.spatial_mode == "auto"
        assert cfg.n_hrus == 1

    def test_default_calibration_params_consistent(self):
        """The documented 14-parameter default set matches cFUSE's bounds."""
        from cfuse import PARAM_BOUNDS
        assert len(DEFAULT_CALIBRATION_PARAMS) == 14
        for name in DEFAULT_CALIBRATION_PARAMS:
            assert name in PARAM_BOUNDS, f"{name} not a valid cFUSE parameter"


class TestValidation:
    def test_invalid_model_structure_rejected(self):
        with pytest.raises(ValidationError):
            CFUSEConfig(model_structure="not_a_real_structure")

    def test_n_hrus_must_be_positive(self):
        with pytest.raises(ValidationError):
            CFUSEConfig(n_hrus=0)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            CFUSEConfig(this_field_does_not_exist=True)

    def test_invalid_calibration_metric_rejected(self):
        with pytest.raises(ValidationError):
            CFUSEConfig(calibration_metric="RMSE")


class TestCalibrationParams:
    def test_get_calibration_params_parses_csv(self):
        cfg = CFUSEConfig(params_to_calibrate="S1_max, ku ,b")
        assert cfg.get_calibration_params() == ["S1_max", "ku", "b"]

    def test_unknown_params_warn_but_do_not_fail(self):
        # The validator logs a warning for unknown names but still accepts them
        # (final validation happens at runtime against the installed cFUSE).
        cfg = CFUSEConfig(params_to_calibrate="S1_max,definitely_not_a_param")
        assert "definitely_not_a_param" in cfg.get_calibration_params()
