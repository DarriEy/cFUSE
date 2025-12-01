#!/usr/bin/env python3
"""Diagnostic script to find source of NaN gradients in dFUSE"""

import numpy as np
import torch
import dfuse_core

# Check what functions are available
print("=== Available dfuse_core functions ===")
funcs = [f for f in dir(dfuse_core) if not f.startswith('_')]
print(funcs)

# Simple test parameters
PARAM_NAMES = [
    'S1_max', 'S2_max', 'f_tens', 'f_rchr', 'f_base', 'r1',
    'ku', 'c', 'alpha', 'psi', 'kappa', 'ki',
    'ks', 'n', 'v', 'v_A', 'v_B',
    'Ac_max', 'b', 'lambda', 'chi', 'mu_t',
    'T_rain', 'T_melt', 'melt_rate', 'lapse_rate', 'opg',
    'MFMAX', 'MFMIN'
]

# Safe default parameters (from DEFAULT_INIT_PARAMS in optimize_basin)
params = np.array([
    100.0,   # S1_max
    1000.0,  # S2_max
    0.5,     # f_tens
    0.5,     # f_rchr
    0.5,     # f_base
    0.5,     # r1
    500.0,   # ku
    10.5,    # c
    125.5,   # alpha
    3.0,     # psi
    0.5,     # kappa
    500.0,   # ki
    50.0,    # ks
    5.0,     # n
    0.125,   # v
    0.125,   # v_A
    0.125,   # v_B
    0.5,     # Ac_max
    1.5,     # b
    7.5,     # lambda
    3.5,     # chi
    0.9,     # mu_t
    1.0,     # T_rain
    1.0,     # T_melt
    5.5,     # melt_rate
    -5.0,    # lapse_rate
    0.5,     # opg
    4.2,     # MFMAX
    2.4,     # MFMIN
], dtype=np.float32)

print(f"\n=== Parameters ({len(params)}) ===")
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name}: {params[i]}")

# Config (VIC-style)
config_dict = {
    'upper_arch': 1,
    'lower_arch': 1,
    'baseflow': 2,
    'percolation': 0,
    'surface_runoff': 1,
    'evaporation': 1,
    'interflow': 0,
    'enable_snow': True,
}

# Try to load actual data
print("\n=== Loading actual data ===")
try:
    from dfuse_netcdf import read_fuse_forcing, read_elevation_bands
    import os
    
    data_dir = os.path.expanduser("~/compHydro/data/FUSE")
    forcing_path = f"{data_dir}/forcing/usgs_14138800_forcing.nc"
    bands_path = f"{data_dir}/settings/usgs_14138800/elevbands_usgs_14138800.nc"
    
    forcing_data = read_fuse_forcing(forcing_path)
    bands = read_elevation_bands(bands_path)
    
    print(f"  Forcing timesteps: {len(forcing_data.time)}")
    print(f"  Elevation bands: {len(bands.area_frac)}")
    
    # Stack forcing
    forcing = np.column_stack([
        forcing_data.precip,
        forcing_data.temp,
        forcing_data.pet
    ]).astype(np.float32)
    
    area_frac = bands.area_frac.astype(np.float32)
    mean_elev = bands.mean_elev.astype(np.float32)
    ref_elev = float(np.sum(area_frac * mean_elev))
    
    print(f"  Forcing shape: {forcing.shape}")
    print(f"  Area fractions: {area_frac}")
    print(f"  Mean elevations: {mean_elev}")
    print(f"  Reference elev: {ref_elev}")
    
    # Check for NaN/Inf in forcing
    print(f"  NaN in forcing: P={np.any(np.isnan(forcing[:,0]))}, T={np.any(np.isnan(forcing[:,1]))}, PET={np.any(np.isnan(forcing[:,2]))}")
    print(f"  Inf in forcing: {np.any(np.isinf(forcing))}")
    
except Exception as e:
    print(f"  Could not load data: {e}")
    # Use synthetic data
    forcing = np.zeros((1000, 3), dtype=np.float32)
    forcing[:, 0] = 5.0    # precip
    forcing[:, 1] = 10.0   # temp
    forcing[:, 2] = 3.0    # PET
    area_frac = np.array([1.0], dtype=np.float32)
    mean_elev = np.array([1500.0], dtype=np.float32)
    ref_elev = 1500.0

# Initial state
state = np.zeros(9 + 30, dtype=np.float32)
state[0] = 50.0   # S1
state[5] = 250.0  # S2

print("\n=== Testing forward pass ===")
n_test = min(1000, len(forcing))
try:
    result = dfuse_core.run_fuse_elevation_bands(
        state, forcing[:n_test], params, config_dict,
        area_frac, mean_elev, ref_elev,
        None, 1.0, False, False, 1, "euler"
    )
    states, runoff, fluxes = result
    print(f"  States shape: {states.shape}")
    print(f"  Runoff shape: {runoff.shape}")
    print(f"  Runoff range: {runoff.min():.6f} - {runoff.max():.6f}")
    print(f"  Runoff mean: {runoff.mean():.6f}")
    print(f"  Any NaN in runoff: {np.any(np.isnan(runoff))}")
    print(f"  Any Inf in runoff: {np.any(np.isinf(runoff))}")
    print(f"  Zero runoff count: {np.sum(runoff == 0)} / {len(runoff)}")
    
    if np.any(np.isnan(runoff)):
        print("  !!! NaN detected in forward pass !!!")
        nan_idx = np.where(np.isnan(runoff))[0]
        print(f"  First NaN at timestep: {nan_idx[0]}")
        print(f"  Runoff before NaN: {runoff[max(0, nan_idx[0]-5):nan_idx[0]]}")
        
except Exception as e:
    print(f"  ERROR in forward pass: {e}")
    import traceback
    traceback.print_exc()
    runoff = None

print("\n=== Testing routing ===")
if runoff is not None:
    try:
        shape = 2.5
        delay = float(params[PARAM_NAMES.index('mu_t')])
        routed = dfuse_core.route_runoff(runoff, shape, delay, 1.0)
        print(f"  Routed shape: {routed.shape}")
        print(f"  Routed range: {routed.min():.6f} - {routed.max():.6f}")
        print(f"  Any NaN in routed: {np.any(np.isnan(routed))}")
    except Exception as e:
        print(f"  ERROR in routing: {e}")

print("\n=== Testing gradient computation ===")
if runoff is not None and not np.any(np.isnan(runoff)):
    try:
        # Simple gradient (ones)
        grad_output = np.ones(n_test, dtype=np.float32)
        
        grad_params = dfuse_core.compute_gradient_adjoint_bands(
            state, forcing[:n_test], params, grad_output,
            config_dict,
            area_frac, mean_elev,
            ref_elev, 2.5, 1.0, 1.0
        )
        
        print(f"  Gradient shape: {grad_params.shape}")
        print(f"  Gradient range: {grad_params.min():.6e} - {grad_params.max():.6e}")
        print(f"  Any NaN in gradient: {np.any(np.isnan(grad_params))}")
        print(f"  Any Inf in gradient: {np.any(np.isinf(grad_params))}")
        
        if np.any(np.isnan(grad_params)):
            print("  !!! NaN detected in gradient !!!")
            nan_idx = np.where(np.isnan(grad_params))[0]
            print(f"  NaN gradient params: {[PARAM_NAMES[i] for i in nan_idx]}")
        else:
            print("\n  Non-zero gradients:")
            for i, name in enumerate(PARAM_NAMES):
                if abs(grad_params[i]) > 1e-10:
                    print(f"    {name}: {grad_params[i]:.6e}")
                    
    except Exception as e:
        print(f"  ERROR in gradient computation: {e}")
        import traceback
        traceback.print_exc()

print("\n=== Testing NSE-like loss gradient ===")
if runoff is not None and not np.any(np.isnan(runoff)):
    try:
        # Create fake observations (perturbed runoff)
        obs = runoff * 1.1 + 0.5
        
        # NSE gradient: d(loss)/d(runoff) = 2 * (sim - obs) / var(obs)
        residuals = runoff - obs
        var_obs = np.var(obs)
        grad_output = (2.0 * residuals / var_obs).astype(np.float32)
        
        print(f"  Grad output range: {grad_output.min():.6e} - {grad_output.max():.6e}")
        print(f"  Any NaN in grad_output: {np.any(np.isnan(grad_output))}")
        
        if not np.any(np.isnan(grad_output)):
            grad_params = dfuse_core.compute_gradient_adjoint_bands(
                state, forcing[:n_test], params, grad_output,
                config_dict,
                area_frac, mean_elev,
                ref_elev, 2.5, 1.0, 1.0
            )
            
            print(f"  Gradient with NSE-like loss:")
            print(f"    Any NaN: {np.any(np.isnan(grad_params))}")
            if not np.any(np.isnan(grad_params)):
                print(f"    Range: {grad_params.min():.6e} - {grad_params.max():.6e}")
            else:
                nan_idx = np.where(np.isnan(grad_params))[0]
                print(f"    NaN params: {[PARAM_NAMES[i] for i in nan_idx]}")
                
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n=== Done ===")
print("\nIf NaN appears in gradient but not in forward pass,")
print("the issue is likely in Enzyme AD or parameter bounds.")
print("Try rebuilding with: make clean && make -j")
