"""
Basin Data Manager for AquaWatch EO
Generates and caches multi-annual 5-year time-series (2021-2025) for:
1. Oder River (Odra - German-Polish Border) - Narrow river channel (120-200m)
2. Lake Constance (Bodensee) - Deep pre-alpine lake (90m depth)
3. Lake Müggelsee (Berlin) - Shallow polymictic lowland lake (4m depth)
4. Bautzen Reservoir (Saxony) - Dammed reservoir with hydraulic controls
5. Bavarian Danube Reach (Donau) - Fast-flowing alpine river reach
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASIN_PROFILES = {
    "oder_river": {
        "name": "Oder River (Odra - German-Polish Border)",
        "type": "River",
        "lat": 52.345,
        "lon": 14.550,
        "state": "Brandenburg (German-Polish Border)",
        "scale_desc": "Narrow River Channel (120–200m width, mean depth 2.8m)",
        "base_summer_temp": 20.8,
        "thermal_inertia": 0.35,  # Low thermal inertia -> fast, flashy heating
        "base_ndvi": -0.06,
        "max_bloom_potency": 0.48, # 2022 disaster saw extreme bloom
        "primary_taxa": "Prymnesium parvum (Golden Alga) & Cyanobacteria",
        "jurisdiction": "German-Polish Transboundary Reach (Brandenburg / Lubuskie)",
        "flow_velocity_mps": 0.25, # Low summer flow
        "base_supply_m3s": 56.0,
        "base_demand_m3s": 22.0,
        "min_ecological_flow_m3s": 18.5,
        "years": {
            2021: {"temp_offset": -0.7, "heat_potency": 1.2, "hot_streak": 2},
            2022: {"temp_offset": +2.2, "heat_potency": 5.2, "hot_streak": 9},  # The 2022 disaster!
            2023: {"temp_offset": +0.8, "heat_potency": 3.4, "hot_streak": 5},
            2024: {"temp_offset": +0.3, "heat_potency": 2.5, "hot_streak": 4},
            2025: {"temp_offset": +1.6, "heat_potency": 4.6, "hot_streak": 7}
        }
    },
    "lake_constance": {
        "name": "Lake Constance (Bodensee)",
        "type": "Lake",
        "lat": 47.636,
        "lon": 9.380,
        "state": "Baden-Württemberg / Bavaria",
        "scale_desc": "Large Pre-Alpine Deep Lake (536 km², mean depth 90m)",
        "base_summer_temp": 18.8,
        "thermal_inertia": 0.85,  # High thermal inertia -> slow, smooth temperature change
        "base_ndvi": -0.08,       # Clear oligotrophic-mesotrophic water
        "max_bloom_potency": 0.26,# Slower, moderate blooms restricted to shallow bays
        "primary_taxa": "Planktothrix rubescens & Diatoms",
        "jurisdiction": "Bodensee-Wasserversorgung (Bavaria / Baden-Württemberg)",
        "flow_velocity_mps": 0.02,
        "base_supply_m3s": 380.0,
        "base_demand_m3s": 175.0,
        "min_ecological_flow_m3s": 190.0,
        "years": {
            2021: {"temp_offset": -0.9, "heat_potency": 0.8, "hot_streak": 1},
            2022: {"temp_offset": +1.3, "heat_potency": 3.6, "hot_streak": 6},
            2023: {"temp_offset": +0.7, "heat_potency": 2.6, "hot_streak": 4},
            2024: {"temp_offset": +0.2, "heat_potency": 1.9, "hot_streak": 3},
            2025: {"temp_offset": +1.2, "heat_potency": 3.4, "hot_streak": 5}
        }
    },
    "lake_mueggelsee": {
        "name": "Lake Müggelsee (Berlin)",
        "type": "Lake",
        "lat": 52.433,
        "lon": 13.630,
        "state": "Berlin",
        "scale_desc": "Shallow Eutrophic Lowland Lake (7.4 km², mean depth 4.0m)",
        "base_summer_temp": 21.5,
        "thermal_inertia": 0.40,
        "base_ndvi": -0.03,       # Eutrophic baseline
        "max_bloom_potency": 0.42, # High cyanobacteria biomass
        "primary_taxa": "Microcystis aeruginosa & Aphanizomenon",
        "jurisdiction": "Berlin Senatsverwaltung für Umwelt / Badegewässer",
        "flow_velocity_mps": 0.05,
        "base_supply_m3s": 11.5,
        "base_demand_m3s": 7.8,
        "min_ecological_flow_m3s": 3.2,
        "years": {
            2021: {"temp_offset": -0.5, "heat_potency": 1.5, "hot_streak": 2},
            2022: {"temp_offset": +1.6, "heat_potency": 4.5, "hot_streak": 8},
            2023: {"temp_offset": +1.0, "heat_potency": 3.8, "hot_streak": 6},
            2024: {"temp_offset": +0.4, "heat_potency": 2.8, "hot_streak": 4},
            2025: {"temp_offset": +1.5, "heat_potency": 4.2, "hot_streak": 7}
        }
    },
    "bautzen_reservoir": {
        "name": "Bautzen Reservoir (Saxony)",
        "type": "Reservoir",
        "lat": 51.215,
        "lon": 14.453,
        "state": "Saxony",
        "scale_desc": "Dammed Reservoir with Bottom Weir (5.3 km², mean depth 7.4m)",
        "base_summer_temp": 20.2,
        "thermal_inertia": 0.50,
        "base_ndvi": -0.05,
        "max_bloom_potency": 0.35,
        "primary_taxa": "Microcystis & Anabaena",
        "jurisdiction": "Landestalsperrenverwaltung Sachsen (LTV)",
        "flow_velocity_mps": 0.08,
        "base_supply_m3s": 24.0,
        "base_demand_m3s": 13.5,
        "min_ecological_flow_m3s": 6.5,
        "years": {
            2021: {"temp_offset": -0.6, "heat_potency": 1.1, "hot_streak": 2},
            2022: {"temp_offset": +1.5, "heat_potency": 4.0, "hot_streak": 7},
            2023: {"temp_offset": +0.8, "heat_potency": 3.1, "hot_streak": 5},
            2024: {"temp_offset": +0.3, "heat_potency": 2.2, "hot_streak": 3},
            2025: {"temp_offset": +1.4, "heat_potency": 3.9, "hot_streak": 6}
        }
    },
    "danube_river": {
        "name": "Bavarian Danube Reach (Donau)",
        "type": "River",
        "lat": 48.950,
        "lon": 12.300,
        "state": "Bavaria (Regensburg / Deggendorf)",
        "scale_desc": "Flowing Alpine River Reach (150m width, mean depth 3.5m)",
        "base_summer_temp": 19.8,
        "thermal_inertia": 0.30,
        "base_ndvi": -0.07,
        "max_bloom_potency": 0.22, # Rapid flushing limits pelagic blooms
        "primary_taxa": "Potamoplankton & Chlorophyceae",
        "jurisdiction": "Wasserwirtschaftsamt Regensburg / Deggendorf",
        "flow_velocity_mps": 0.65, # Higher velocity
        "base_supply_m3s": 440.0,
        "base_demand_m3s": 150.0,
        "min_ecological_flow_m3s": 240.0,
        "years": {
            2021: {"temp_offset": -0.8, "heat_potency": 0.9, "hot_streak": 1},
            2022: {"temp_offset": +1.2, "heat_potency": 3.2, "hot_streak": 5},
            2023: {"temp_offset": +0.6, "heat_potency": 2.4, "hot_streak": 3},
            2024: {"temp_offset": +0.2, "heat_potency": 1.8, "hot_streak": 2},
            2025: {"temp_offset": +1.1, "heat_potency": 3.0, "hot_streak": 4}
        }
    },
    "lake_starnberg": {
        "name": "Lake Starnberg (Starnberger See)",
        "type": "Lake",
        "lat": 47.900,
        "lon": 11.310,
        "state": "Bavaria (near Munich)",
        "scale_desc": "Deep Pre-Alpine Monomictic Lake (56 km², mean depth 53m)",
        "base_summer_temp": 19.4,
        "thermal_inertia": 0.82,
        "base_ndvi": -0.075,
        "max_bloom_potency": 0.24,
        "primary_taxa": "Planktothrix rubescens & Diatoms",
        "jurisdiction": "Wasserwirtschaftsamt Weilheim / Bayerisches LfU",
        "flow_velocity_mps": 0.015,
        "base_supply_m3s": 18.0,
        "base_demand_m3s": 8.5,
        "min_ecological_flow_m3s": 6.0,
        "years": {
            2021: {"temp_offset": -0.8, "heat_potency": 0.8, "hot_streak": 1},
            2022: {"temp_offset": +1.4, "heat_potency": 3.4, "hot_streak": 5},
            2023: {"temp_offset": +0.7, "heat_potency": 2.5, "hot_streak": 3},
            2024: {"temp_offset": +0.2, "heat_potency": 1.7, "hot_streak": 2},
            2025: {"temp_offset": +1.3, "heat_potency": 3.2, "hot_streak": 5}
        }
    },
    "lake_chiemsee": {
        "name": "Lake Chiemsee ('Bavarian Sea')",
        "type": "Lake",
        "lat": 47.880,
        "lon": 12.400,
        "state": "Bavaria ('Bavarian Sea')",
        "scale_desc": "Large Pre-Alpine Lowland Lake (80 km², mean depth 26m)",
        "base_summer_temp": 20.3,
        "thermal_inertia": 0.65,
        "base_ndvi": -0.065,
        "max_bloom_potency": 0.29,
        "primary_taxa": "Anabaena & Microcystis",
        "jurisdiction": "Wasserwirtschaftsamt Rosenheim / Traunstein",
        "flow_velocity_mps": 0.03,
        "base_supply_m3s": 35.0,
        "base_demand_m3s": 14.0,
        "min_ecological_flow_m3s": 12.0,
        "years": {
            2021: {"temp_offset": -0.7, "heat_potency": 1.0, "hot_streak": 2},
            2022: {"temp_offset": +1.5, "heat_potency": 3.8, "hot_streak": 6},
            2023: {"temp_offset": +0.8, "heat_potency": 2.8, "hot_streak": 4},
            2024: {"temp_offset": +0.3, "heat_potency": 2.0, "hot_streak": 3},
            2025: {"temp_offset": +1.4, "heat_potency": 3.6, "hot_streak": 5}
        }
    },
    "rhine_river": {
        "name": "Middle Rhine Reach (Mittelrhein / Kaub)",
        "type": "River",
        "lat": 50.150,
        "lon": 7.720,
        "state": "Rhineland-Palatinate / Hesse",
        "scale_desc": "Major European Inland Waterway (300m width, mean depth 3.2m)",
        "base_summer_temp": 21.2,
        "thermal_inertia": 0.38,
        "base_ndvi": -0.07,
        "max_bloom_potency": 0.20,
        "primary_taxa": "Potamoplankton & Diatoms",
        "jurisdiction": "Wasserstraßen- und Schifffahrtsamt Rhein (WSA)",
        "flow_velocity_mps": 1.10,
        "base_supply_m3s": 1450.0,
        "base_demand_m3s": 420.0,
        "min_ecological_flow_m3s": 780.0,
        "years": {
            2021: {"temp_offset": -0.6, "heat_potency": 1.0, "hot_streak": 2},
            2022: {"temp_offset": +1.7, "heat_potency": 4.2, "hot_streak": 7},
            2023: {"temp_offset": +0.9, "heat_potency": 3.2, "hot_streak": 4},
            2024: {"temp_offset": +0.4, "heat_potency": 2.3, "hot_streak": 3},
            2025: {"temp_offset": +1.5, "heat_potency": 4.0, "hot_streak": 6}
        }
    },
    "elbe_river": {
        "name": "Middle Elbe Reach (Magdeburg)",
        "type": "River",
        "lat": 52.130,
        "lon": 11.630,
        "state": "Saxony-Anhalt",
        "scale_desc": "Lowland Sandbed River with Groyne Fields (220m width, mean depth 2.2m)",
        "base_summer_temp": 21.8,
        "thermal_inertia": 0.32,
        "base_ndvi": -0.05,
        "max_bloom_potency": 0.38,
        "primary_taxa": "Cyanobacteria & Microcystis in Groyne Basins",
        "jurisdiction": "Wasserstraßen- und Schifffahrtsamt Elbe",
        "flow_velocity_mps": 0.30,
        "base_supply_m3s": 210.0,
        "base_demand_m3s": 85.0,
        "min_ecological_flow_m3s": 65.0,
        "years": {
            2021: {"temp_offset": -0.6, "heat_potency": 1.2, "hot_streak": 2},
            2022: {"temp_offset": +1.9, "heat_potency": 4.8, "hot_streak": 8},
            2023: {"temp_offset": +0.8, "heat_potency": 3.3, "hot_streak": 5},
            2024: {"temp_offset": +0.3, "heat_potency": 2.4, "hot_streak": 3},
            2025: {"temp_offset": +1.5, "heat_potency": 4.3, "hot_streak": 7}
        }
    },
    "steinhuder_meer": {
        "name": "Lake Steinhude (Steinhuder Meer)",
        "type": "Lake",
        "lat": 52.480,
        "lon": 9.320,
        "state": "Lower Saxony",
        "scale_desc": "Shallow Polymictic Lowland Lake (29 km², mean depth 1.4m)",
        "base_summer_temp": 22.0,
        "thermal_inertia": 0.25,
        "base_ndvi": -0.02,
        "max_bloom_potency": 0.44,
        "primary_taxa": "Microcystis & Planktothrix agardhii",
        "jurisdiction": "Niedersächsischer Landesbetrieb für Wasserwirtschaft (NLWKN)",
        "flow_velocity_mps": 0.02,
        "base_supply_m3s": 6.5,
        "base_demand_m3s": 4.8,
        "min_ecological_flow_m3s": 1.8,
        "years": {
            2021: {"temp_offset": -0.5, "heat_potency": 1.4, "hot_streak": 2},
            2022: {"temp_offset": +1.8, "heat_potency": 4.6, "hot_streak": 8},
            2023: {"temp_offset": +1.0, "heat_potency": 3.6, "hot_streak": 5},
            2024: {"temp_offset": +0.4, "heat_potency": 2.6, "hot_streak": 3},
            2025: {"temp_offset": +1.6, "heat_potency": 4.4, "hot_streak": 7}
        }
    }
}


def generate_basin_timeseries(basin_key: str, out_dir: str = "data/basins") -> pd.DataFrame:
    """Generate 5-year summer dataset (2021-2025, 460 daily records) tailored to basin morphology."""
    profile = BASIN_PROFILES.get(basin_key, BASIN_PROFILES["oder_river"])
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f"{basin_key}_5years.csv")

    dfs = []
    total_days = 92  # June 1 to August 31

    for yr in range(2021, 2026):
        start_dt = datetime(yr, 6, 1)
        dates = [start_dt + timedelta(days=i) for i in range(total_days)]
        yr_cfg = profile["years"].get(yr, {"temp_offset": 0.0, "heat_potency": 2.5, "hot_streak": 4})
        
        seed = int(sum(ord(c) for c in basin_key) + yr * 7) % 50000
        np.random.seed(seed)
        
        day_indices = np.arange(total_days)
        # Seasonal solar cycle
        seasonal_wave = np.sin(np.pi * (day_indices / total_days)) * 4.2
        
        # Heatwave peak timing
        hw_center = int(total_days * (0.52 + 0.08 * np.sin(yr)))
        hw_shape = yr_cfg["heat_potency"] * np.exp(-((day_indices - hw_center) ** 2) / (2 * (5.5 ** 2)))
        
        noise = np.random.normal(0, 0.35, total_days)
        base_t = profile["base_summer_temp"] + yr_cfg["temp_offset"]
        lst_series = base_t + seasonal_wave + hw_shape + noise

        # River vs Lake optical signature
        ndwi_base = 0.55 if profile["type"] == "River" else 0.62
        ndwi_series = ndwi_base + np.random.normal(0, 0.015, total_days)

        ndvi_series = np.full(total_days, profile["base_ndvi"])
        fai_series = np.full(total_days, profile["base_ndvi"] * 0.3)

        consecutive_hot = 0
        thresh = 22.0 if profile["type"] == "Lake" else 21.5
        for i in range(total_days):
            temp = lst_series[i]
            if temp >= thresh:
                consecutive_hot += 1
            else:
                consecutive_hot = max(0, consecutive_hot - 1)

            if consecutive_hot >= 3:
                # River flushing damping vs Lake stagnation amplification
                flushing_damping = 0.75 if profile["flow_velocity_mps"] > 0.4 else 1.25
                bloom_factor = min(1.0, (consecutive_hot - 2) * 0.18 + (temp - thresh) * 0.08) * flushing_damping
                ndvi_val = profile["base_ndvi"] + bloom_factor * profile["max_bloom_potency"] + np.random.normal(0, 0.015)
                fai_val = (profile["base_ndvi"] * 0.3) + bloom_factor * (profile["max_bloom_potency"] * 0.22) + np.random.normal(0, 0.004)
            else:
                ndvi_val = profile["base_ndvi"] + np.random.normal(0, 0.01)
                fai_val = (profile["base_ndvi"] * 0.3) + np.random.normal(0, 0.003)

            ndvi_series[i] = round(float(ndvi_val), 4)
            fai_series[i] = round(float(fai_val), 4)

        # Hydrological modeling: Coupled Water Supply & Water Demand
        base_supply = profile.get("base_supply_m3s", 55.0)
        base_demand = profile.get("base_demand_m3s", 22.0)
        min_eco_flow = profile.get("min_ecological_flow_m3s", 18.0)

        cumulative_heat = np.cumsum(np.maximum(0.0, lst_series - thresh))
        drought_factor = np.clip(
            0.42 * (hw_shape / (yr_cfg["heat_potency"] + 1e-4)) + 
            0.38 * (cumulative_heat / (total_days * 3.2)),
            0.0, 0.65
        )
        supply_noise = np.random.normal(0, 0.02, total_days)
        supply_series = np.maximum(
            min_eco_flow,
            base_supply * (1.0 - drought_factor) * (1.0 + supply_noise)
        )

        temp_excess = np.maximum(0.0, lst_series - 19.0)
        demand_surge = (temp_excess / 8.0) * 0.70
        demand_noise = np.random.normal(0, 0.02, total_days)
        demand_series = base_demand * (1.0 + demand_surge) * (1.0 + demand_noise)

        # Algae Formation Intensity Index (0 to 100%)
        algae_idx_series = np.clip(
            (ndvi_series - profile["base_ndvi"]) / (profile["max_bloom_potency"] - profile["base_ndvi"] + 1e-4) * 100.0,
            0.0, 100.0
        )

        df_yr = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "year": yr,
            "basin_key": basin_key,
            "basin_name": profile["name"],
            "basin_type": profile["type"],
            "lst_c": [round(float(v), 2) for v in lst_series],
            "water_supply_m3s": [round(float(v), 1) for v in supply_series],
            "water_demand_m3s": [round(float(v), 1) for v in demand_series],
            "ndvi": ndvi_series,
            "algae_formation_pct": [round(float(v), 1) for v in algae_idx_series],
            "ndwi": [round(float(v), 4) for v in ndwi_series],
            "fai": fai_series
        })
        dfs.append(df_yr)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(cache_path, index=False)
    return combined_df
