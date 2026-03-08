"""
cFUSE - Differentiable FUSE Hydrological Model

A differentiable implementation of the FUSE (Framework for Understanding 
Structural Errors) hydrological model with Enzyme automatic differentiation.

Example usage:
    from cfuse import FUSEConfig, VIC_CONFIG
    from cfuse.io import read_fuse_forcing, read_elevation_bands
"""

__version__ = "0.4.1"

# Import core configuration
from .config import (
    FUSEConfig,
    VIC_CONFIG,
    TOPMODEL_CONFIG,
    ARNO_CONFIG,
    PRMS_CONFIG,
    SACRAMENTO_CONFIG,
    PARAM_NAMES,
    PARAM_BOUNDS,
    DEFAULT_PARAMS,
    get_default_params_array,
    UpperLayerArch,
    LowerLayerArch,
    BaseflowType,
    PercolationType,
    SurfaceRunoffType,
    EvaporationType,
    InterflowType,
)

# Import I/O utilities
from .io import (
    read_fuse_forcing,
    read_elevation_bands,
    parse_file_manager,
    parse_fuse_decisions,
    FUSEForcing,
    FUSEElevationBands,
)

# Try to import the compiled C++ core
try:
    import cfuse_core as core
    HAS_CORE = True
    HAS_ENZYME = getattr(core, 'HAS_ENZYME', False)
except ImportError:
    HAS_CORE = False
    HAS_ENZYME = False
    core = None

__all__ = [
    # Version
    "__version__",
    # Configuration classes
    "FUSEConfig",
    "UpperLayerArch",
    "LowerLayerArch",
    "BaseflowType",
    "PercolationType",
    "SurfaceRunoffType",
    "EvaporationType",
    "InterflowType",
    # Preset configs
    "VIC_CONFIG",
    "TOPMODEL_CONFIG",
    "ARNO_CONFIG",
    "PRMS_CONFIG",
    "SACRAMENTO_CONFIG",
    # Parameters
    "PARAM_NAMES",
    "PARAM_BOUNDS",
    "DEFAULT_PARAMS",
    "get_default_params_array",
    # I/O
    "read_fuse_forcing",
    "read_elevation_bands",
    "parse_file_manager",
    "parse_fuse_decisions",
    "FUSEForcing",
    "FUSEElevationBands",
    # Core
    "core",
    "HAS_CORE",
    "HAS_ENZYME",
    # SYMFLUENCE plugin
    "register",
]


def register():
    """Register cFUSE as a SYMFLUENCE plugin.

    Called automatically when installed as a SYMFLUENCE entry point,
    or can be called manually to register cFUSE components with
    the SYMFLUENCE model and optimizer registries.
    """
    from symfluence.core.registry import model_manifest
    from symfluence.models.registry import ModelRegistry
    from symfluence.optimization.registry import OptimizerRegistry
    from symfluence.cli.services import BuildInstructionsRegistry

    from .sfconfig import CFUSEConfigAdapter
    from .extractor import CFUSEResultExtractor
    from .runner import CFUSERunner
    from .preprocessor import CFUSEPreProcessor
    from .postprocessor import CFUSEPostprocessor, CFUSERoutedPostprocessor
    from .build_instructions import get_cfuse_build_instructions
    from .calibration.optimizer import CFUSEModelOptimizer
    from .calibration.worker import CFUSEWorker
    from .calibration.parameter_manager import CFUSEParameterManager
    from .calibration.targets import CFUSEStreamflowTarget

    # Register via model_manifest (config adapter + result extractor)
    model_manifest(
        "CFUSE",
        config_adapter=CFUSEConfigAdapter,
        result_extractor=CFUSEResultExtractor,
        build_instructions_module="cfuse.build_instructions",
    )

    # Register runner, preprocessor, postprocessors
    ModelRegistry.register_runner('CFUSE', method_name='run_cfuse')(CFUSERunner)
    ModelRegistry.register_preprocessor('CFUSE')(CFUSEPreProcessor)
    ModelRegistry.register_postprocessor('CFUSE')(CFUSEPostprocessor)
    ModelRegistry.register_postprocessor('CFUSE_routed')(CFUSERoutedPostprocessor)

    # Register build instructions
    BuildInstructionsRegistry.register('cfuse')(get_cfuse_build_instructions)

    # Register calibration components
    OptimizerRegistry.register_optimizer('CFUSE')(CFUSEModelOptimizer)
    OptimizerRegistry.register_worker('CFUSE')(CFUSEWorker)
    OptimizerRegistry.register_parameter_manager('CFUSE')(CFUSEParameterManager)
    OptimizerRegistry.register_calibration_target('CFUSE', 'streamflow')(CFUSEStreamflowTarget)
