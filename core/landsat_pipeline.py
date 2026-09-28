"""
Landsat Earth Observation Data Pipeline
Handles:
1. Microsoft Planetary Computer & USGS Landsat Collection 2 Level 2 STAC queries
2. Pre-configured real-world freshwater benchmark basins (Lake Balaton, Western Lake Erie, Lake Victoria, Bodensee, Danube Reach)
3. Realistic high-fidelity Landsat 8/9 Level 2 scene synthesis with realistic bathymetry, water boundaries, and thermal fields
4. GeoJSON / Raster export for GIS and water authority decision systems
"""

import os
import json
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta


# Pre-defined Benchmark Freshwater Basins representing varied scales & climates
BENCHMARK_BASINS = {
    "lake_balaton": {
        "name": "Lake Balaton (Keszthely & Siofok Basins)",
        "country": "Hungary, Central Europe",
        "bbox": [17.15, 46.70, 18.25, 47.10],  # [min_lon, min_lat, max_lon, max_lat]
        "center": [46.85, 17.75],
        "scale_type": "large_shallow_lake",
        "area_km2": 592.0,
        "mean_depth_m": 3.2,
        "primary_bloom_taxa": "Cylindrospermopsis raciborskii & Microcystis",
        "baseline_summer_temp_c": 23.5,
        "typical_heatwave_temp_c": 27.2,
        "description": "Central Europe's largest lake. Extremely shallow and sensitive to summer heatwaves and western basin nutrient concentration."
    },
    "lake_erie_west": {
        "name": "Western Lake Erie Basin",
        "country": "USA / Canada",
        "bbox": [-83.55, 41.35, -82.50, 42.10],
        "center": [41.75, -83.05],
        "scale_type": "large_lake_embayment",
        "area_km2": 4800.0,
        "mean_depth_m": 7.4,
        "primary_bloom_taxa": "Microcystis aeruginosa",
        "baseline_summer_temp_c": 22.0,
        "typical_heatwave_temp_c": 26.5,
        "description": "Global benchmark for toxic cyanobacterial blooms threatening municipal drinking water intakes (Maumee River discharge zone)."
    },
    "lake_victoria_winam": {
        "name": "Lake Victoria (Winam Gulf)",
        "country": "Kenya / Uganda / Tanzania",
        "bbox": [34.20, -0.60, 34.85, -0.05],
        "center": [-0.30, 34.50],
        "scale_type": "tropical_semi_enclosed_gulf",
        "area_km2": 1400.0,
        "mean_depth_m": 6.0,
        "primary_bloom_taxa": "Microcystis & Anabaena flos-aquae",
        "baseline_summer_temp_c": 26.5,
        "typical_heatwave_temp_c": 29.8,
        "description": "Hyper-eutrophic tropical embayment with persistent high temperatures and massive surface scum proliferation."
    },
    "lake_constance": {
        "name": "Lake Constance (Bodensee)",
        "country": "Germany / Switzerland / Austria",
        "bbox": [9.10, 47.50, 9.75, 47.75],
        "center": [47.63, 9.40],
        "scale_type": "deep_alpine_lake",
        "area_km2": 536.0,
        "mean_depth_m": 90.0,
        "primary_bloom_taxa": "Planktothrix rubescens (Burgundy blood)",
        "baseline_summer_temp_c": 18.5,
        "typical_heatwave_temp_c": 23.0,
        "description": "Deep, oligotrophic alpine benchmark. Clear open water with isolated warm embayments prone to autumn overturn blooms."
    },
    "oder_river": {
        "name": "Oder River (Odra - German-Polish Border)",
        "country": "Germany (Brandenburg) / Poland",
        "bbox": [14.40, 52.20, 14.75, 52.65],
        "center": [52.42, 14.58],
        "scale_type": "narrow_river_channel",
        "area_km2": 28.5,
        "mean_depth_m": 2.8,
        "primary_bloom_taxa": "Prymnesium parvum (Golden Alga) & Cyanobacteria",
        "baseline_summer_temp_c": 21.5,
        "typical_heatwave_temp_c": 26.8,
        "description": "Historic European transboundary hotspot (August 2022 disaster). Narrow 120-200m channel requiring sub-pixel pure-water masking and low-flow heatwave tracking."
    },
    "danube_river_reach": {
        "name": "Danube River Reach & Impoundment",
        "country": "Central / Southeastern Europe",
        "bbox": [18.85, 45.60, 19.30, 45.95],
        "center": [45.78, 19.05],
        "scale_type": "river_channel_and_weir",
        "area_km2": 45.0,
        "mean_depth_m": 4.5,
        "primary_bloom_taxa": "Potamoplankton & Chlorophyceae",
        "baseline_summer_temp_c": 21.0,
        "typical_heatwave_temp_c": 25.0,
        "description": "Narrow, flowing channel with slow-moving impoundments. Tests 30m/100m shoreline mixed-pixel filtering."
    }
}


class LandsatPipeline:
    """
    Automated acquisition and data generator for Landsat 8/9 Level-2 freshwater analysis.
    """

    def __init__(self, data_cache_dir: str = "data/basins"):
        self.cache_dir = data_cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def query_planetary_computer_stac(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Query Microsoft Planetary Computer STAC API for Landsat 8/9 Level 2 scenes.
        """
        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        payload = {
            "collections": ["landsat-c2-l2"],
            "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "query": {
                "eo:cloud_cover": {"lt": max_cloud_cover},
                "platform": {"in": ["landsat-8", "landsat-9"]}
            },
            "limit": 10
        }
        try:
            resp = requests.post(stac_url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                features = data.get("features", [])
                results = []
                for feat in features:
                    props = feat.get("properties", {})
                    results.append({
                        "id": feat.get("id"),
                        "datetime": props.get("datetime"),
                        "cloud_cover": props.get("eo:cloud_cover"),
                        "platform": props.get("platform"),
                        "assets": list(feat.get("assets", {}).keys())
                    })
                return results
            else:
                return []
        except Exception:
            # Fallback for offline or network-restricted environments
            return []

    def generate_benchmark_basin_scene(
        self,
        basin_key: str = "lake_balaton",
        heatwave_condition: bool = False,
        grid_size: Tuple[int, int] = (120, 160)
    ) -> Dict[str, Any]:
        """
        Generate high-fidelity Landsat 8/9 Level 2 multi-spectral scene for a benchmark basin.
        Generates realistic Blue, Green, Red, NIR, SWIR1, SWIR2, and ST_B10 (Thermal) bands.
        """
        meta = BENCHMARK_BASINS.get(basin_key, BENCHMARK_BASINS["lake_balaton"])
        h, w = grid_size
        y, x = np.mgrid[0:h, 0:w]
        
        # Normalized coordinates [-1, 1]
        nx = (x / (w - 1)) * 2.0 - 1.0
        ny = (y / (h - 1)) * 2.0 - 1.0

        # Construct realistic aquatic geometry based on basin type
        if "river" in meta["scale_type"]:
            # Sinusoidal meandering river channel
            channel_center = 0.4 * np.sin(ny * 3.5)
            dist_to_river = np.abs(nx - channel_center)
            water_mask = dist_to_river < 0.14
            # Add an impoundment/lake widening in the middle
            widening = ((ny + 0.1)**2 + (nx - 0.2)**2) < 0.12
            water_mask = water_mask | widening
        elif "embayment" in meta["scale_type"] or "gulf" in meta["scale_type"]:
            # Semi-enclosed bay / gulf with irregular coastline
            bay_shape = (nx + 0.3)**2 / 0.8**2 + (ny - 0.1)**2 / 0.6**2
            noise = 0.08 * np.sin(nx * 10.0) + 0.05 * np.cos(ny * 8.0)
            water_mask = (bay_shape + noise) < 0.70
        else:
            # Elongated lake (e.g. Balaton, Constance)
            lake_shape = (nx * 0.7)**2 / 0.75**2 + (ny * 1.6)**2 / 0.55**2
            noise = 0.06 * np.sin(nx * 8.0) + 0.04 * np.cos(ny * 6.0)
            water_mask = (lake_shape + noise) < 0.65

        # Land baseline reflectance (soil + terrestrial vegetation)
        # Land has high NIR (0.35-0.55), moderate Red (0.08-0.18), low Blue/Green
        land_nir = 0.42 + 0.10 * np.sin(nx * 3.0) + np.random.normal(0, 0.02, (h, w))
        land_red = 0.12 + 0.04 * np.cos(ny * 2.5) + np.random.normal(0, 0.01, (h, w))
        land_green = 0.10 + 0.03 * np.sin(ny * 2.0) + np.random.normal(0, 0.01, (h, w))
        land_blue = 0.06 + 0.02 * np.cos(nx * 2.0) + np.random.normal(0, 0.01, (h, w))
        land_swir1 = 0.22 + 0.05 * np.sin(nx * 2.0) + np.random.normal(0, 0.02, (h, w))
        land_swir2 = 0.14 + 0.04 * np.cos(ny * 2.0) + np.random.normal(0, 0.01, (h, w))
        land_temp = 31.0 + 3.0 * np.sin(nx * 2.0) + np.random.normal(0, 0.5, (h, w))
        if heatwave_condition:
            land_temp += 4.5

        # Water baseline reflectance (clean water strongly absorbs NIR/SWIR)
        water_blue = 0.05 + np.random.normal(0, 0.003, (h, w))
        water_green = 0.04 + np.random.normal(0, 0.003, (h, w))
        water_red = 0.02 + np.random.normal(0, 0.002, (h, w))
        water_nir = 0.008 + np.random.normal(0, 0.001, (h, w))
        water_swir1 = 0.003 + np.random.normal(0, 0.001, (h, w))
        water_swir2 = 0.001 + np.random.normal(0, 0.001, (h, w))

        # Base water surface temperature (°C)
        base_temp = meta["typical_heatwave_temp_c"] if heatwave_condition else meta["baseline_summer_temp_c"]
        # Spatial temperature gradient: shallow western/bay zones warm faster by 2-3°C
        thermal_gradient = (nx * -1.5) + (ny * 0.8)
        water_temp = base_temp + thermal_gradient + np.random.normal(0, 0.35, (h, w))

        # Algal Bloom Spatial Simulation:
        # Blooms cluster in warm, sheltered, nutrient-rich bays (western side of Balaton / western Lake Erie)
        bloom_center_x = -0.35
        bloom_center_y = 0.15
        dist_from_bloom_epicenter = np.sqrt((nx - bloom_center_x)**2 + (ny - bloom_center_y)**2)
        
        # Bloom intensity is amplified under heatwave conditions
        bloom_potency = 1.8 if heatwave_condition else 0.85
        bloom_intensity = np.exp(-dist_from_bloom_epicenter / 0.45) * bloom_potency
        bloom_intensity = np.clip(bloom_intensity + np.random.normal(0, 0.06, (h, w)), 0.0, 1.0)
        
        # When algae bloom occurs:
        # NIR reflectance surges (chlorophyll cellular scattering), Green increases, Red dips or absorbs
        water_nir = np.where(water_mask, water_nir + bloom_intensity * 0.18, water_nir)
        water_green = np.where(water_mask, water_green + bloom_intensity * 0.08, water_green)
        water_red = np.where(water_mask, water_red + bloom_intensity * 0.04, water_red)

        # Shoreline Adjacency Effect simulation:
        # Edge pixels have mixed signal from land vegetation (high NIR)
        from scipy import ndimage
        eroded_water = ndimage.binary_erosion(water_mask, iterations=2)
        shoreline_edges = water_mask & (~eroded_water)
        
        # Bleed 25% of land signal into shoreline water pixels
        water_nir[shoreline_edges] += 0.09
        water_red[shoreline_edges] += 0.03
        water_temp[shoreline_edges] += 1.8  # Bank soil warms edge water

        # Composite final scene bands
        final_blue = np.where(water_mask, water_blue, land_blue)
        final_green = np.where(water_mask, water_green, land_green)
        final_red = np.where(water_mask, water_red, land_red)
        final_nir = np.where(water_mask, water_nir, land_nir)
        final_swir1 = np.where(water_mask, water_swir1, land_swir1)
        final_swir2 = np.where(water_mask, water_swir2, land_swir2)
        final_temp = np.where(water_mask, water_temp, land_temp)

        # Convert temperature to Landsat Level 2 ST_B10 raw Digital Numbers
        # ST_K = (temp_c + 273.15)
        # DN = (ST_K - 149.0) / 0.00341802
        st_kelvin = final_temp + 273.15
        raw_b10_dn = ((st_kelvin - 149.0) / 0.00341802).astype(np.uint16)

        # Time series thermal history (past 21 days for lag & heatwave analysis)
        historical_days = 21
        historical_dates = [(datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(historical_days - 1, -1, -1)]
        
        if heatwave_condition:
            # Heatwave curve: steadily rises past 23C and stays high for 6 days
            history_curve = np.linspace(20.5, 27.5, historical_days) + np.random.normal(0, 0.4, historical_days)
            history_curve[-6:] = 26.5 + np.random.normal(0, 0.3, 6)
        else:
            history_curve = base_temp + 1.2 * np.sin(np.linspace(0, 3.14, historical_days)) + np.random.normal(0, 0.5, historical_days)

        return {
            "metadata": meta,
            "basin_key": basin_key,
            "scene_id": f"LC09_L2SP_{basin_key.upper()}_{datetime.now().strftime('%Y%m%d')}",
            "acquisition_date": datetime.now().strftime("%Y-%m-%d"),
            "is_heatwave_scenario": heatwave_condition,
            "bands": {
                "blue": final_blue,
                "green": final_green,
                "red": final_red,
                "nir": final_nir,
                "swir1": final_swir1,
                "swir2": final_swir2,
                "thermal": raw_b10_dn,
                "lst_celsius_ground_truth": final_temp
            },
            "water_mask_ground_truth": water_mask,
            "historical_dates": historical_dates,
            "historical_mean_lst": history_curve.tolist()
        }
