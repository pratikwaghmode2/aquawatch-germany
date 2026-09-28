"""
Comprehensive Unit & Integration Tests for Landsat Algae Bloom Pipeline
"""

import pytest
import numpy as np
import pandas as pd
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
from core.resolution_adapter import ResolutionAdapter, FreshwaterSystemScale
from core.thermal_dynamics import ThermalDynamicsAnalyzer
from core.ml_models import BloomMachineLearningEngine
from core.forecasting_engine import BloomForecastingEngine
from core.landsat_pipeline import LandsatPipeline, BENCHMARK_BASINS


def test_water_indices():
    green = np.array([0.15, 0.05])
    swir1 = np.array([0.05, 0.20])
    mndwi = calculate_mndwi(green, swir1)
    
    # Water pixel should have positive MNDWI (0.15 > 0.05 -> 0.10/0.20 = 0.5)
    assert mndwi[0] > 0.0
    # Land pixel should have negative MNDWI (0.05 < 0.20 -> -0.15/0.25 = -0.6)
    assert mndwi[1] < 0.0

    nir = np.array([0.02, 0.35])
    swir2 = np.array([0.01, 0.15])
    awei = calculate_awei(green, swir1, nir, swir2)
    assert awei[0] > awei[1]

    # Test FAI
    fai = calculate_fai(nir, green, swir1)
    assert fai.shape == (2,)

    # Test NDCI
    red = np.array([0.01, 0.10])
    ndci = calculate_ndci(nir, red)
    assert -1.0 <= ndci[0] <= 1.0


def test_lst_conversion():
    # Test Kelvin to Celsius
    raw_kelvin = np.array([293.15, 300.15])  # 20C and 27C
    celsius = calculate_lst_celsius(raw_kelvin)
    assert np.isclose(celsius[0], 20.0, atol=0.1)
    assert np.isclose(celsius[1], 27.0, atol=0.1)

    # Test USGS Collection 2 Level 2 Digital Numbers
    # ST_K = DN * 0.00341802 + 149.0
    # For 295.15 K (22 C): (295.15 - 149.0) / 0.00341802 = 42758.67
    dn = np.array([42759])
    celsius_from_dn = calculate_lst_celsius(dn)
    assert np.isclose(celsius_from_dn[0], 22.0, atol=0.5)


def test_resolution_adapter():
    adapter = ResolutionAdapter(pixel_resolution_m=30.0)
    
    # Create synthetic lake mask (40x40 circle)
    y, x = np.ogrid[-20:20, -20:20]
    water_mask = (x*x + y*y) <= 15*15
    
    stats = adapter.classify_system_scale(water_mask)
    assert "scale" in stats
    assert stats["water_pixel_count"] > 0
    assert stats["surface_area_km2"] > 0

    pure_mask, shoreline_mask = adapter.extract_pure_water_mask(water_mask, iterations=1)
    assert np.sum(pure_mask) < np.sum(water_mask)
    assert np.sum(shoreline_mask) > 0
    assert np.sum(pure_mask & shoreline_mask) == 0


def test_thermal_dynamics():
    analyzer = ThermalDynamicsAnalyzer()
    
    # 7-day temp sequence with 4 consecutive hot days >= 22C
    temps = np.array([19.0, 20.5, 23.0, 24.5, 25.0, 23.5, 18.0])
    exceedance = analyzer.calculate_consecutive_exceedance_days(temps, threshold_c=22.0)
    assert exceedance[0] == 0
    assert exceedance[1] == 0
    assert exceedance[2] == 1
    assert exceedance[3] == 2
    assert exceedance[4] == 3
    assert exceedance[5] == 4
    assert exceedance[6] == 0  # reset on drop

    cdd = analyzer.calculate_cumulative_degree_days(temps, base_temp_c=20.0)
    assert cdd[-1] > 0.0

    # Lagged bloom response test
    bloom_series = np.roll(temps, 3) * 0.05 + 0.1
    lag_res = analyzer.analyze_lagged_bloom_response(temps, bloom_series, max_lag_days=5)
    assert "optimal_lag_days" in lag_res
    assert "lag_correlations" in lag_res


def test_ml_pipeline_and_forecasting():
    pipeline = LandsatPipeline()
    scene = pipeline.generate_benchmark_basin_scene("lake_balaton", heatwave_condition=True, grid_size=(30, 40))
    
    assert "bands" in scene
    assert "blue" in scene["bands"]
    assert "thermal" in scene["bands"]

    metrics = compute_comprehensive_water_metrics(scene["bands"])
    assert "fai" in metrics
    assert "water_mask" in metrics

    ml_engine = BloomMachineLearningEngine(random_state=42)
    train_metrics = ml_engine.train()
    assert train_metrics["test_f1_score"] > 0.70
    assert len(ml_engine.feature_importance_) == len(BloomMachineLearningEngine.FEATURE_NAMES)

    # Test Forecasting Engine
    forecaster = BloomForecastingEngine(ml_engine)
    h, w = scene["bands"]["blue"].shape
    dummy_features = np.zeros((h, w, 10))
    dummy_features[:, :, 0] = 24.0  # current temp
    dummy_features[:, :, 3] = 4     # 4 hot days
    
    forecast_res = forecaster.simulate_spatial_forecast(
        base_features=dummy_features,
        water_mask=metrics["water_mask"],
        forecast_days=7,
        temperature_delta_c=2.5,
        consecutive_exceedance_days=6,
        threshold_c=22.0
    )
    assert "hazard_tier_grid" in forecast_res
    assert "trajectory" in forecast_res
    assert "water_authority_advisory" in forecast_res
    assert forecast_res["water_authority_advisory"]["alert_code"] in ["GREEN_NORMAL", "YELLOW_WATCH", "ORANGE_WARNING", "RED_ALERT"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
