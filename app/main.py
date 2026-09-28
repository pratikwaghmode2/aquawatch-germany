"""
FastAPI Application for Freshwater Algae Bloom Earth Observation & Forecasting
Serves the interactive Water Authority Decision Support Dashboard and REST API.
"""

import os
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.landsat_pipeline import LandsatPipeline, BENCHMARK_BASINS
from core.water_indices import (
    compute_comprehensive_water_metrics,
    classify_bloom_hazard_tier
)
from core.resolution_adapter import ResolutionAdapter
from core.thermal_dynamics import ThermalDynamicsAnalyzer
from core.ml_models import BloomMachineLearningEngine
from core.forecasting_engine import BloomForecastingEngine


app = FastAPI(
    title="Freshwater EO Algal Bloom & Thermal Early Warning System",
    description="Operational Landsat 8/9 Level-2 Earth Observation and Machine Learning forecasting for water authorities.",
    version="1.0.0"
)

# Initialize engines
pipeline = LandsatPipeline()
res_adapter = ResolutionAdapter(pixel_resolution_m=30.0)
thermal_analyzer = ThermalDynamicsAnalyzer()
ml_engine = BloomMachineLearningEngine()
ml_engine.train()
forecaster = BloomForecastingEngine(ml_engine)

# Cache loaded basin analyses to serve rapid dashboard requests
basin_cache: Dict[str, Dict[str, Any]] = {}


def process_basin(basin_key: str, heatwave: bool = False) -> Dict[str, Any]:
    """Process a benchmark basin through the full EO and thermal pipeline."""
    cache_id = f"{basin_key}_{heatwave}"
    if cache_id in basin_cache:
        return basin_cache[cache_id]

    scene = pipeline.generate_benchmark_basin_scene(basin_key, heatwave_condition=heatwave, grid_size=(80, 100))
    meta = scene["metadata"]
    bands = scene["bands"]
    
    # 1. Compute indices
    indices = compute_comprehensive_water_metrics(bands)
    water_mask = indices["water_mask"]
    
    # 2. System scale & shoreline buffering
    scale_info = res_adapter.classify_system_scale(water_mask)
    pure_mask, shoreline_mask = res_adapter.extract_pure_water_mask(water_mask)
    
    # 3. Adjacency & thermal corrections
    corrected_fai = res_adapter.correct_shoreline_adjacency(indices["fai"], water_mask, pure_mask)
    corrected_lst = res_adapter.harmonize_thermal_resolution(indices["lst_celsius"], water_mask, pure_mask)
    
    # 4. Bloom Hazard Classification
    hazard_grid, tier_percentages = classify_bloom_hazard_tier(indices["bloom_severity_index"], corrected_lst)
    
    # 5. Thermal dynamics & Lag analysis
    hist_lst = np.array(scene["historical_mean_lst"])
    thermal_state = thermal_analyzer.compute_lake_thermal_state(corrected_lst, hist_lst, threshold_c=22.0)
    
    # Generate synthetic historical bloom series correlated with temp
    hist_bloom = np.roll(hist_lst, 4) * 0.04 - 0.5 + np.random.normal(0, 0.03, len(hist_lst))
    hist_bloom = np.clip(hist_bloom, 0.05, 0.95)
    lag_analysis = thermal_analyzer.analyze_lagged_bloom_response(hist_lst, hist_bloom, max_lag_days=10)
    
    # 6. Feature stack for ML & forecasting
    h, w = water_mask.shape
    features = np.zeros((h, w, 10))
    features[:, :, 0] = np.nan_to_num(corrected_lst, nan=20.0)
    features[:, :, 1] = features[:, :, 0] - 0.4
    features[:, :, 2] = features[:, :, 0] - 0.8
    features[:, :, 3] = thermal_state["consecutive_days_above_threshold"]
    features[:, :, 4] = thermal_state["14_day_cdd_above_20c"]
    features[:, :, 5] = features[:, :, 0] - meta["baseline_summer_temp_c"]
    # Turbidity NDTI
    red = bands["red"]
    green = bands["green"]
    features[:, :, 6] = np.clip((red - green) / (red + green + 1e-6), -0.5, 0.5)
    features[:, :, 7] = 2 if "large" in scale_info["scale_name"] else (1 if "medium" in scale_info["scale_name"] else 0)
    features[:, :, 8] = 0.5
    features[:, :, 9] = 0.85

    result = {
        "metadata": meta,
        "basin_key": basin_key,
        "scene_id": scene["scene_id"],
        "acquisition_date": scene["acquisition_date"],
        "is_heatwave": heatwave,
        "scale_info": scale_info,
        "thermal_state": thermal_state,
        "lag_analysis": lag_analysis,
        "historical_dates": scene["historical_dates"],
        "historical_mean_lst": scene["historical_mean_lst"],
        "tier_percentages": tier_percentages,
        "feature_shape": [h, w],
        # Serialized 2D arrays downsampled for fast geo-json/canvas rendering
        "grids": {
            "water_mask": water_mask.astype(int).tolist(),
            "pure_water_mask": pure_mask.astype(int).tolist(),
            "shoreline_mask": shoreline_mask.astype(int).tolist(),
            "lst_celsius": [[round(float(v), 1) if not np.isnan(v) else None for v in row] for row in corrected_lst],
            "fai": [[round(float(v), 3) if not np.isnan(v) else None for v in row] for row in corrected_fai],
            "bloom_severity_index": [[round(float(v), 3) if not np.isnan(v) else None for v in row] for row in indices["bloom_severity_index"]],
            "hazard_tier": hazard_grid.tolist()
        },
        "_raw_features": features,
        "_water_mask": water_mask
    }
    basin_cache[cache_id] = result
    return result


class ForecastRequest(BaseModel):
    basin_key: str = "lake_balaton"
    heatwave_base: bool = False
    temperature_delta_c: float = 2.0
    consecutive_exceedance_days: int = 5
    threshold_c: float = 22.0
    forecast_horizon_days: int = 7
    wind_mixing: str = "stagnant"
    trophic_state: str = "eutrophic"


@app.get("/api/basins")
def list_basins():
    """List available benchmark freshwater basins with geographic metadata."""
    return {
        "basins": [
            {
                "key": key,
                "name": val["name"],
                "country": val["country"],
                "center": val["center"],
                "bbox": val["bbox"],
                "area_km2": val["area_km2"],
                "primary_taxa": val["primary_bloom_taxa"],
                "scale_type": val["scale_type"],
                "description": val["description"]
            }
            for key, val in BENCHMARK_BASINS.items()
        ]
    }


@app.get("/api/basin/{basin_key}")
def get_basin_data(basin_key: str, heatwave: bool = False):
    """Retrieve full Landsat analysis, thermal diagnostics, and spatial grids."""
    if basin_key not in BENCHMARK_BASINS:
        return JSONResponse(status_code=404, content={"error": f"Basin '{basin_key}' not found."})
    
    data = process_basin(basin_key, heatwave=heatwave)
    # Return JSON without private numpy objects
    resp = {k: v for k, v in data.items() if not k.startswith("_")}
    return resp


@app.post("/api/forecast")
def run_forecast(req: ForecastRequest):
    """Run dynamic scenario forecasting under user-specified temperature and duration pre-conditions."""
    if req.basin_key not in BENCHMARK_BASINS:
        return JSONResponse(status_code=404, content={"error": f"Basin '{req.basin_key}' not found."})

    base_data = process_basin(req.basin_key, heatwave=req.heatwave_base)
    base_features = base_data["_raw_features"]
    water_mask = base_data["_water_mask"]

    forecast_output = forecaster.simulate_spatial_forecast(
        base_features=base_features,
        water_mask=water_mask,
        forecast_days=req.forecast_horizon_days,
        temperature_delta_c=req.temperature_delta_c,
        consecutive_exceedance_days=req.consecutive_exceedance_days,
        threshold_c=req.threshold_c,
        wind_mixing=req.wind_mixing,
        trophic_state=req.trophic_state
    )

    # Serialize grids
    hazard_grid = forecast_output["hazard_tier_grid"].tolist()
    bsi_grid = [
        [round(float(v), 3) if not np.isnan(v) else None for v in row]
        for row in forecast_output["bloom_severity_grid"]
    ]

    return {
        "basin_key": req.basin_key,
        "scenario_pre_conditions": forecast_output["scenario_pre_conditions"],
        "tier_distribution_pct": forecast_output["tier_distribution_pct"],
        "mean_projected_bsi": forecast_output["mean_projected_bsi"],
        "trajectory": forecast_output["trajectory"],
        "water_authority_advisory": forecast_output["water_authority_advisory"],
        "grids": {
            "forecast_hazard_tier": hazard_grid,
            "forecast_bsi": bsi_grid
        }
    }


@app.get("/api/ml-insights")
def get_ml_insights():
    """Retrieve ML training metrics, feature importance ranking, and limnological insights."""
    return {
        "metrics": ml_engine.metrics_,
        "feature_importance": ml_engine.feature_importance_,
        "feature_descriptions": {
            "lst_current": "Current Landsat Lake Surface Water Temperature (LSWT in °C)",
            "lst_3d_mean": "3-day moving thermal average (short-term warming momentum)",
            "lst_7d_mean": "7-day moving thermal average (stratification persistence)",
            "consecutive_hot_days": "Number of consecutive days >= critical threshold",
            "cdd_20": "Cumulative Thermal Degree Days above 20°C (thermal growth fuel)",
            "thermal_anomaly": "Departure from 30-day seasonal baseline",
            "turbidity_index": "Normalized Difference Turbidity Index (suspended sediment/clarity)",
            "system_scale_code": "Freshwater classification (0=River/Small, 1=Reservoir, 2=Large Lake)",
            "shoreline_dist_norm": "Distance to shoreline (embayment shelter vs open pelagic)",
            "solar_insolation_factor": "Seasonal clear-sky solar angle"
        }
    }


@app.get("/api/stac-query")
def stac_query(
    min_lon: float = Query(17.15),
    min_lat: float = Query(46.70),
    max_lon: float = Query(18.25),
    max_lat: float = Query(47.10),
    start_date: str = Query("2026-06-01"),
    end_date: str = Query("2026-09-28"),
    max_cloud: float = Query(25.0)
):
    """Query Planetary Computer STAC for live Landsat 8/9 scenes."""
    bbox = [min_lon, min_lat, max_lon, max_lat]
    scenes = pipeline.query_planetary_computer_stac(bbox, start_date, end_date, max_cloud_cover=max_cloud)
    return {
        "bbox": bbox,
        "query_period": f"{start_date} to {end_date}",
        "scenes_found": len(scenes),
        "scenes": scenes
    }


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serve the complete interactive Water Authority Earth Observation Dashboard."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
