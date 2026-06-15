#!/usr/bin/env python3
"""
Tests for the FUSE text-file parsers in cfuse.netcdf.

These parse legacy Fortran-FUSE file-manager and decision files. They are pure
text parsers (no NetCDF / symfluence needed) and previously had no coverage.
"""

from textwrap import dedent

from cfuse.netcdf import parse_file_manager, parse_fuse_decisions


class TestParseFileManager:
    def test_maps_quoted_values_in_order(self, tmp_path):
        fm_file = tmp_path / "fm.txt"
        fm_file.write_text(dedent("""\
            FUSE_FILEMANAGER_V1.0
            ! this is a comment, skipped
            '/data/settings/'   ! setngs_path
            '/data/input/'      ! input_path
            '/data/output/'     ! output_path
            '_forcing.nc'       ! suffix_forcing
            '_elev_bands.nc'    ! suffix_elev_bands
        """))
        fm = parse_file_manager(fm_file)
        assert fm["setngs_path"] == "/data/settings/"
        assert fm["input_path"] == "/data/input/"
        assert fm["output_path"] == "/data/output/"
        assert fm["suffix_forcing"] == "_forcing.nc"
        assert fm["suffix_elev_bands"] == "_elev_bands.nc"

    def test_skips_comments_and_header(self, tmp_path):
        fm_file = tmp_path / "fm.txt"
        fm_file.write_text(dedent("""\
            FUSE_FILEMANAGER_V1.0
            ! comment line
            '/only/one/'   ! setngs_path
        """))
        fm = parse_file_manager(fm_file)
        # Only one value parsed; header/comment lines must not consume a key slot.
        assert fm == {"setngs_path": "/only/one/"}


class TestParseFuseDecisions:
    def test_maps_decision_names_to_attributes(self, tmp_path):
        dec_file = tmp_path / "decisions.txt"
        dec_file.write_text(dedent("""\
            ! FUSE decisions
            additive_e   RFERR    ! rainfall error
            tension2_1   ARCH1    ! upper-layer architecture
            unlimfrc_2   ARCH2    ! lower-layer architecture
            arno_x_vic   QSURF    ! surface runoff
            perc_w2sat   QPERC    ! percolation
            sequential   ESOIL    ! evaporation
            intflwsome   QINTF    ! interflow
            rout_gamma   Q_TDH    ! routing
            temp_index   SNOWM    ! snow model
            0 (end of decisions)
            ignored_after_break  ARCH1
        """))
        d = parse_fuse_decisions(dec_file)
        assert d.arch1 == "tension2_1"
        assert d.arch2 == "unlimfrc_2"
        assert d.qsurf == "arno_x_vic"
        assert d.qperc == "perc_w2sat"
        assert d.esoil == "sequential"
        assert d.qintf == "intflwsome"
        assert d.q_tdh == "rout_gamma"
        assert d.snowmod == "temp_index"

    def test_stops_at_break_line(self, tmp_path):
        """A line beginning with '0' terminates parsing."""
        dec_file = tmp_path / "decisions.txt"
        dec_file.write_text(dedent("""\
            tension2_1   ARCH1
            0 (break)
            should_not_win   ARCH1
        """))
        d = parse_fuse_decisions(dec_file)
        assert d.arch1 == "tension2_1"
