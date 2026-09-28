"""
Core package for Landsat Earth Observation Algal Bloom & Thermal Dynamics
"""

from core.water_indices import (
    calculate_mndwi,
    calculate_awei,
    extract_water_mask,
    calculate_fai,
    calculate_ndci,
    calculate_sabi,
    calculate_lst_celsius,
    compute_comprehensive_water_metrics,
    classify_bloom_hazard_tier
)

from core.resolution_adapter import (
    FreshwaterSystemScale,
    ResolutionAdapter
)

from core.thermal_dynamics import (
    ThermalDynamicsAnalyzer
)

from core.ml_models import (
    BloomMachineLearningEngine
)

from core.forecasting_engine import (
    BloomForecastingEngine
)

from core.landsat_pipeline import (
    BENCHMARK_BASINS,
    LandsatPipeline
)

__all__ = [
    "calculate_mndwi",
    "calculate_awei",
    "extract_water_mask",
    "calculate_fai",
    "calculate_ndci",
    "calculate_sabi",
    "calculate_lst_celsius",
    "compute_comprehensive_water_metrics",
    "classify_bloom_hazard_tier",
    "FreshwaterSystemScale",
    "ResolutionAdapter",
    "ThermalDynamicsAnalyzer",
    "BloomMachineLearningEngine",
    "BloomForecastingEngine",
    "BENCHMARK_BASINS",
    "LandsatPipeline"
]
