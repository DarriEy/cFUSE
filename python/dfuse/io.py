"""
dFUSE I/O Utilities

Functions for reading FUSE forcing data, elevation bands, and configuration files.
"""

import sys
from pathlib import Path

# Import from the original dfuse_netcdf module (in parent directory for backwards compat)
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from dfuse_netcdf import (
    FUSEForcing,
    FUSEElevationBands,
    FUSEDecisions,
    FortranParameters,
    read_fuse_forcing,
    read_elevation_bands,
    parse_file_manager,
    parse_fuse_decisions,
    parse_fortran_constraints,
    write_fuse_output,
    FUSERunner,
)

__all__ = [
    "FUSEForcing",
    "FUSEElevationBands", 
    "FUSEDecisions",
    "FortranParameters",
    "read_fuse_forcing",
    "read_elevation_bands",
    "parse_file_manager",
    "parse_fuse_decisions",
    "parse_fortran_constraints",
    "write_fuse_output",
    "FUSERunner",
]
