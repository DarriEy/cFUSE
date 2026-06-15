# Changelog

All notable changes to this project will be documented in this file.

## [0.6.1] - 2026-06-15
- Relicense under GPL-3.0-or-later to match the original FUSE implementation;
  update all per-file SPDX headers (previously Apache-2.0) to match.
- Fix parameter vector length (29 → 31): `optimize_basin.py` now imports
  `PARAM_NAMES`/`PARAM_BOUNDS` from `cfuse.config`, and `to_cfuse_params`
  emits `shape_t`/`smooth_frac`. The C++ core now rejects short parameter
  vectors instead of silently using defaults.
- Fix uninitialized `Flux` members and the snow-state derivative in the
  implicit/explicit Euler solver paths (NaN/SWE-corruption hazard); the
  `TENSION2_FREE` evaporation split now matches the differentiable kernel.
- Fix the deprecated pure-Python fallback snow melt to use the degree-day
  factor and `T_melt` base temperature (was misusing `T_melt`/`T_rain`); align
  its `lapse_rate` bounds with `cfuse.config`.
- Fix truncated/over-long NetCDF text attributes (`nc_put_att_text` was passed
  hand-counted lengths) via a `strlen`-based helper.
- Guard the CLI NSE metric against an empty `q_obs` vector (out-of-bounds read)
  and a zero observation variance (division by zero).
- Guard upper/lower-layer evaporation against a near-zero tension capacity
  (`S1_T_max`/`S2_T_max`) that could produce NaN.
- Remove dead code (`compute_final_flux`, `incomplete_gamma`).
- Align the ODE-solver percolation source term with the differentiable kernel
  (shared `percolation_source` helper) so the CLI/`run_fuse` path and the
  gradient path agree for lower-zone-demand percolation with nonlinear baseflow.
- Replace `mktime`/`gmtime` date conversions (timezone-dependent, DST-sensitive,
  not thread-safe) with pure integer civil-date arithmetic.
- Declare `pandas`, `xarray`, and `pydantic` as runtime dependencies.
- Synchronize the version string (0.6.0) across all build metadata.
- CI now runs the full `pytest` suite; add tests for the Fortran parameter
  conversion, parameter-layout sync, the calibration parameter manager, the
  SYMFLUENCE config schema, the FUSE file-manager / decisions parsers, and the
  result extractor.

## [0.6.0] - 2026-06-10
- Require Python >=3.11; drop 3.9/3.10 classifiers.
- Recognize CF-convention forcing variable names in the preprocessor.
- Stabilize the implicit Euler Jacobian and make CI reproducible.
- Adopt the SYMFLUENCE `*PostProcessor` spelling (pre-1.0 alias removal).
- Migrate plugin registration to the unified `R` / `model_manifest` API.

## [0.5.0] - 2026-03-08
- Add SYMFLUENCE plugin integration (`cfuse:register` entry point).

## [0.4.1] - 2024-12-28
- Fix optimize CLI import for packaged installs.
- Rebase example data paths when release file manager uses absolute paths.
- Pass 3D forcing arrays to batch API for stability.
- Package `optimize_basin` as a top-level module for `cfuse-optimize`.

## [0.4.0] - 2024-12-28
- Rename project/package to cFUSE (cfuse/cfuse_core).
- Switch to MIT license and update metadata.
- Move example data to GitHub release assets; README documents the download path.
- Update the Python optimization workflow to use batch APIs and routed gradients.
- Add a Python batch smoke test to the CMake test suite.
- Add `route_runoff` binding in `cfuse_core`.
- Normalize Python package layout (`cfuse.netcdf`, `cfuse.torch`, legacy model module).
- Add CMake-based Python build (`setup.py`), `MANIFEST.in`, and a build-dist workflow.
