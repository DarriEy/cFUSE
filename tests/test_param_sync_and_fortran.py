#!/usr/bin/env python3
"""
Regression tests for the parameter-vector layout.

These guard the 29-vs-31 parameter bug: the C++ core expects
``NUM_PARAMETERS`` (31) values, but several Python producers had drifted to a
stale 29-entry layout that silently dropped ``shape_t`` and ``smooth_frac``.
"""

import warnings

import numpy as np
import pytest

import cfuse_core
from cfuse.config import PARAM_NAMES, DEFAULT_PARAMS
from cfuse.netcdf import FortranParameters


# =============================================================================
# Fortran -> cFUSE parameter conversion
# =============================================================================

class TestFortranParams:
    def test_length_matches_core(self):
        """to_cfuse_params must emit exactly NUM_PARAMETERS values."""
        params = FortranParameters().to_cfuse_params()
        assert params.shape == (cfuse_core.NUM_PARAMETERS,)
        assert params.shape == (len(PARAM_NAMES),)
        assert params.shape == (31,)

    def test_includes_shape_t_and_smooth_frac(self):
        """The two trailing params that used to be dropped are populated."""
        params = FortranParameters().to_cfuse_params()
        shape_t_idx = PARAM_NAMES.index("shape_t")
        smooth_frac_idx = PARAM_NAMES.index("smooth_frac")
        # Defaults sourced from cfuse.config.DEFAULT_PARAMS.
        assert params[shape_t_idx] == pytest.approx(DEFAULT_PARAMS["shape_t"])
        assert params[smooth_frac_idx] == pytest.approx(DEFAULT_PARAMS["smooth_frac"])

    def test_known_mappings(self):
        """A few Fortran->cFUSE field mappings stay stable."""
        fp = FortranParameters()
        params = fp.to_cfuse_params()
        assert params[PARAM_NAMES.index("S1_max")] == pytest.approx(fp.MAXWATR_1)
        assert params[PARAM_NAMES.index("MFMAX")] == pytest.approx(fp.MFMAX)
        assert params[PARAM_NAMES.index("MFMIN")] == pytest.approx(fp.MFMIN)
        assert params[PARAM_NAMES.index("opg")] == pytest.approx(fp.OPG)

    def test_unlimited_lower_arch_uses_large_s2(self):
        """'unlim*' architectures push S2_max effectively to infinity."""
        params = FortranParameters().to_cfuse_params(arch2="unlimpow_2")
        assert params[PARAM_NAMES.index("S2_max")] >= 1e9

    def test_dfuse_alias_matches(self):
        """The backward-compat alias returns the same vector."""
        fp = FortranParameters()
        np.testing.assert_array_equal(fp.to_dfuse_params(), fp.to_cfuse_params())

    def test_core_rejects_short_vector(self):
        """The C++ core must reject a too-short parameter vector loudly."""
        cfg = {
            "upper_arch": 0, "lower_arch": 1, "baseflow": 2, "percolation": 1,
            "surface_runoff": 1, "evaporation": 1, "interflow": 1, "enable_snow": True,
        }
        states = np.zeros((1, 2), dtype=np.float32)
        forcing = np.ones((5, 1, 3), dtype=np.float32)
        short = np.full((29,), 0.5, dtype=np.float32)
        with pytest.raises(Exception):
            cfuse_core.run_fuse_batch(states, forcing, short, cfg, 1.0)


# =============================================================================
# optimize_basin must stay in sync with the canonical layout
# =============================================================================

class TestParamSync:
    def test_optimize_basin_uses_canonical_names(self):
        """optimize_basin imports PARAM_NAMES from cfuse.config; no local drift."""
        import optimize_basin
        assert list(optimize_basin.PARAM_NAMES) == list(PARAM_NAMES)
        assert len(optimize_basin.PARAM_NAMES) == cfuse_core.NUM_PARAMETERS

    def test_every_param_has_bounds_and_default(self):
        from cfuse.config import PARAM_BOUNDS
        for name in PARAM_NAMES:
            assert name in PARAM_BOUNDS, f"{name} missing bounds"
            assert name in DEFAULT_PARAMS, f"{name} missing default"


# =============================================================================
# Deprecated pure-Python fallback: degree-day snow melt
# =============================================================================

class TestLegacySnowFallback:
    def _fallback(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from cfuse.legacy import _run_fuse_python
        return _run_fuse_python

    def test_snowpack_melts_above_base_temp(self):
        """Melt uses melt_rate*(T - T_melt); a warming series should drain SWE."""
        run = self._fallback()
        p = np.zeros(27)
        p[0], p[1] = 200.0, 2000.0       # S1_max, S2_max
        p[6], p[7] = 10.0, 4.0           # ku, c
        p[12], p[13] = 1.0, 2.0          # ks, n
        p[18] = 1.5                      # b
        p[22], p[23], p[24] = 1.0, 0.0, 5.0   # T_rain, T_melt, melt_rate
        T = 60
        forcing = np.stack(
            [np.full(T, 5.0), np.full(T, 2.0), np.linspace(-5.0, 12.0, T)], axis=1
        )
        state = np.array([100.0, 500.0, 30.0])  # start with 30 mm SWE
        final_state, runoff = run(state, forcing, p, {"enable_snow": True}, 1.0)
        assert np.isfinite(runoff).all()
        assert final_state[-1] < 30.0          # snow melted as it warmed
        assert runoff.sum() > 0.0

    def test_frozen_series_retains_snow(self):
        """With temps always below T_melt, SWE should accumulate, not melt away."""
        run = self._fallback()
        p = np.zeros(27)
        p[0], p[1] = 200.0, 2000.0
        p[6], p[7] = 10.0, 4.0
        p[12], p[13] = 1.0, 2.0
        p[18] = 1.5
        p[22], p[23], p[24] = 1.0, 0.0, 5.0
        T = 40
        forcing = np.stack(
            [np.full(T, 4.0), np.full(T, 1.0), np.full(T, -10.0)], axis=1
        )
        state = np.array([100.0, 500.0, 5.0])
        final_state, _ = run(state, forcing, p, {"enable_snow": True}, 1.0)
        assert final_state[-1] > 5.0           # accumulated snow, no spurious melt
