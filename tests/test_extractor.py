#!/usr/bin/env python3
"""
Tests for cfuse.extractor.CFUSEResultExtractor.

The extractor subclasses a SYMFLUENCE base class, so these tests skip when
symfluence is unavailable. They cover the cFUSE-specific extraction logic
(variable-name mapping, CSV/NetCDF extraction, spatial reduction).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("symfluence")

from cfuse.extractor import CFUSEResultExtractor  # noqa: E402


@pytest.fixture
def extractor():
    return CFUSEResultExtractor()


class TestMetadata:
    def test_variable_name_mapping(self, extractor):
        assert extractor.get_variable_names("streamflow") == [
            "streamflow", "discharge", "q_routed", "Q"
        ]
        assert "swe" in extractor.get_variable_names("snow")

    def test_unknown_variable_falls_back_to_itself(self, extractor):
        assert extractor.get_variable_names("mystery") == ["mystery"]

    def test_unit_conversion_flags(self, extractor):
        assert extractor.requires_unit_conversion("runoff") is True
        assert extractor.requires_unit_conversion("streamflow") is False

    def test_spatial_aggregation_method(self, extractor):
        assert extractor.get_spatial_aggregation_method("runoff") == "selection"

    def test_output_file_patterns(self, extractor):
        patterns = extractor.get_output_file_patterns()
        assert "streamflow" in patterns and "runoff" in patterns
        assert any(p.endswith(".csv") for p in patterns["streamflow"])


class TestCsvExtraction:
    def _write_csv(self, path, columns):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"datetime": idx, **columns})
        df.to_csv(path, index=False)

    def test_extract_named_column(self, extractor, tmp_path):
        f = tmp_path / "a_cfuse_output.csv"
        self._write_csv(f, {"streamflow": np.arange(5.0)})
        series = extractor.extract_variable(f, "streamflow")
        assert list(series.values) == [0, 1, 2, 3, 4]

    def test_streamflow_cms_fallback(self, extractor, tmp_path):
        f = tmp_path / "b_cfuse_output.csv"
        self._write_csv(f, {"streamflow_cms": np.arange(5.0) + 10})
        series = extractor.extract_variable(f, "streamflow")
        assert list(series.values) == [10, 11, 12, 13, 14]

    def test_missing_variable_raises(self, extractor, tmp_path):
        f = tmp_path / "c_cfuse_output.csv"
        self._write_csv(f, {"unrelated": np.arange(5.0)})
        with pytest.raises(ValueError):
            extractor.extract_variable(f, "streamflow")


class TestNetcdfExtraction:
    def test_extract_reduces_hru_dimension(self, extractor, tmp_path):
        xr = pytest.importorskip("xarray")
        time = pd.date_range("2020-01-01", periods=4, freq="D")
        # Two HRUs; extractor should select the first.
        data = np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0]])
        ds = xr.Dataset(
            {"streamflow": (("time", "hru"), data)},
            coords={"time": time, "hru": [0, 1]},
        )
        f = tmp_path / "d_cfuse_output.nc"
        ds.to_netcdf(f)
        series = extractor.extract_variable(f, "streamflow")
        assert list(series.values) == [1.0, 2.0, 3.0, 4.0]
