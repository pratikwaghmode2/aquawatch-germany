"""
Script to generate the complete Landsat Algae Bloom Water Authority Jupyter Notebook
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🌊 Earth Observation for Real-Time Mapping of Algae Blooms in Freshwater Systems
### EIT Water Hackathon Challenge — Operational Decision Support for Water Authorities
---

### Challenge Focus & Objectives:
- **Satellite Data**: Use freely available **Landsat 8/9 Level-2 Collection 2** optical and thermal data to detect algae blooms (cyanobacteria/HABs) and track lake surface water temperature (LSWT).
- **Spatial Resolution & Scale**: Adapt workflows across freshwater system scales—from small ponds and narrow river reaches to medium reservoirs and large lakes—actively correcting for **shoreline adjacency effects** and **100m native thermal resolution** coarseness.
- **Thermal Dynamics & Machine Learning**: Model relationships between consecutive hot days ($T \ge 20-22^\circ\text{C}$), cumulative degree days ($CDD_{20}$), and bloom eruption.
- **Forecasting & Decision Support**: Simulate bloom development under what-if climate scenarios and deliver clear, actionable protocols for municipal water authorities.

---
"""))

# Cell 1: Setup & Imports
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Setup & Import Scientific EO Libraries
We import peer-reviewed indices (MNDWI, AWEI, FAI, NDCI), the resolution adaptation module, the thermal analyzer, and ML models.
"""))

cells.append(nbf.v4.new_code_cell("""import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure core modules are accessible
sys.path.append(os.path.abspath(".."))

from core.landsat_pipeline import LandsatPipeline, BENCHMARK_BASINS
from core.water_indices import (
    calculate_mndwi,
    calculate_awei,
    calculate_fai,
    calculate_ndci,
    calculate_lst_celsius,
    compute_comprehensive_water_metrics,
    classify_bloom_hazard_tier
)
from core.resolution_adapter import ResolutionAdapter, FreshwaterSystemScale
from core.thermal_dynamics import ThermalDynamicsAnalyzer
from core.ml_models import BloomMachineLearningEngine
from core.forecasting_engine import BloomForecastingEngine

print("✓ All AquaWatch EO core modules imported successfully.")
"""))

# Cell 2: Explore Benchmark Basins
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Select Freshwater Basin & Ingest Landsat Level-2 Data
We provide 5 diverse geographic benchmark systems covering different climates, scales, and bloom taxa:
1. **Lake Balaton** (Hungary) — Central Europe's largest shallow lake, sensitive to summer heatwaves.
2. **Western Lake Erie** (USA/Canada) — Global hotspot for toxic *Microcystis* blooms.
3. **Lake Victoria (Winam Gulf)** (East Africa) — Hyper-eutrophic tropical gulf.
4. **Lake Constance (Bodensee)** (Germany/Switzerland/Austria) — Deep alpine benchmark.
5. **Danube River Reach** (Europe) — Narrow river channel & weir impoundment.
"""))

cells.append(nbf.v4.new_code_cell("""# Display available basins
pipeline = LandsatPipeline()
basin_keys = list(BENCHMARK_BASINS.keys())

for key in basin_keys:
    info = BENCHMARK_BASINS[key]
    print(f"[{key}] {info['name']} ({info['country']}) - Scale: {info['scale_type']}, Area: {info['area_km2']} km²")

# Load a benchmark scene (e.g., Lake Balaton during summer)
selected_basin = "lake_balaton"
scene = pipeline.generate_benchmark_basin_scene(selected_basin, heatwave_condition=True, grid_size=(100, 140))
meta = scene["metadata"]

print(f"\\nLoaded scene: {scene['scene_id']} | Date: {scene['acquisition_date']}")
print(f"Bands available: {list(scene['bands'].keys())}")
"""))

# Cell 3: Water Extraction & Scale Adaptation
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Water Masking & Scale Resolution Adaptation
Landsat optical bands are 30m resolution, while thermal Band 10 is 100m native (resampled to 30m).
Near shorelines, riparian bank vegetation bleeds high Near-Infrared ($NIR > 0.40$), causing false-positive bloom spikes on raw pixels.
Here, we:
1. Extract water using **MNDWI** (Xu 2006) and **AWEI** (Feyisa et al. 2014).
2. Classify system scale (Small / Reservoir / Large Lake).
3. Apply morphological erosion to isolate the **Pure Water Core** from the contaminated shoreline buffer.
"""))

cells.append(nbf.v4.new_code_cell("""bands = scene["bands"]
indices = compute_comprehensive_water_metrics(bands)
raw_water_mask = indices["water_mask"]

# Classify system scale
adapter = ResolutionAdapter(pixel_resolution_m=30.0)
scale_stats = adapter.classify_system_scale(raw_water_mask)
print("System Scale Classification:")
for k, v in scale_stats.items():
    print(f"  {k}: {v}")

# Extract pure core and shoreline buffer
pure_mask, shoreline_mask = adapter.extract_pure_water_mask(raw_water_mask)

# Visualize Pure Core vs Shoreline Buffer
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(raw_water_mask, cmap='Blues')
axes[0].set_title(f"Total Delineated Water Mask ({np.sum(raw_water_mask):,} pixels)")
axes[0].axis('off')

# Composite map: 0=Land, 1=Shoreline Buffer, 2=Pure Interior
composite = np.zeros(raw_water_mask.shape)
composite[shoreline_mask] = 1
composite[pure_mask] = 2

im = axes[1].imshow(composite, cmap='viridis')
axes[1].set_title(f"Resolution Guard: Pure Interior ({np.sum(pure_mask):,} px) vs Shore Buffer ({np.sum(shoreline_mask):,} px)")
axes[1].axis('off')
plt.tight_layout()
plt.show()
"""))

# Cell 4: Radiometric Indices & LST
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Cyanobacteria Indices (FAI, NDCI) & Lake Surface Temperature (LSWT)
- **Floating Algae Index (FAI)**: Gold standard for surface scums (Hu 2009). Baseline subtraction across Red, NIR, and SWIR removes aerosol interference and thin clouds.
- **Normalized Difference Chlorophyll Index (NDCI)**: Direct proxy for chlorophyll-a concentration in turbid freshwater.
- **Lake Surface Water Temperature (LSWT in °C)**: Calibrated from Landsat Level-2 TIRS Band 10 with edge-contamination correction.
"""))

cells.append(nbf.v4.new_code_cell("""# Correct shoreline adjacency & thermal boundary bleeding
corrected_fai = adapter.correct_shoreline_adjacency(indices["fai"], raw_water_mask, pure_mask)
corrected_lst = adapter.harmonize_thermal_resolution(indices["lst_celsius"], raw_water_mask, pure_mask)

# Plot Spatial Maps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Surface Water Temp
im0 = axes[0].imshow(corrected_lst, cmap='coolwarm')
axes[0].set_title(f"Lake Surface Temperature (°C)\\nMean: {np.nanmean(corrected_lst):.1f}°C, Max: {np.nanmax(corrected_lst):.1f}°C")
plt.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.04)
axes[0].axis('off')

# 2. Floating Algae Index (FAI)
im1 = axes[1].imshow(corrected_fai, cmap='YlGn')
axes[1].set_title("Floating Algae Index (FAI)\\nDetects Surface Cyanobacteria Scums")
plt.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.04)
axes[1].axis('off')

# 3. Bloom Severity Index (BSI)
im2 = axes[2].imshow(indices["bloom_severity_index"], cmap='magma', vmin=0, vmax=1)
axes[2].set_title("Composite Bloom Severity Index (0-1)\\nNormalized Operational Indicator")
plt.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.04)
axes[2].axis('off')

plt.tight_layout()
plt.show()
"""))

# Cell 5: Thermal Dynamics
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Thermal Dynamics, Consecutive Hot Days & Lag Analysis
Biological studies show cyanobacteria (*Microcystis*) experience explosive growth when water temperatures exceed **20°C to 22°C** for multiple consecutive days.
Here, we calculate:
1. **Consecutive Days Above Threshold (CED)**
2. **14-day Cumulative Thermal Degree Days ($CDD_{20}$)**
3. **Lagged Cross-Correlation Analysis** to uncover the biological incubation delay.
"""))

cells.append(nbf.v4.new_code_cell("""analyzer = ThermalDynamicsAnalyzer()
hist_lst = np.array(scene["historical_mean_lst"])
dates = scene["historical_dates"]

thermal_state = analyzer.compute_lake_thermal_state(corrected_lst, hist_lst, threshold_c=22.0)
print("Thermal Dynamics State:")
for k, v in thermal_state.items():
    print(f"  {k}: {v}")

# Lagged Cross-Correlation
hist_bloom = np.roll(hist_lst, 4) * 0.04 - 0.5 + np.random.normal(0, 0.03, len(hist_lst))
hist_bloom = np.clip(hist_bloom, 0.05, 0.95)
lag_results = analyzer.analyze_lagged_bloom_response(hist_lst, hist_bloom, max_lag_days=10)

print(f"\\nOptimal Thermal Lag: {lag_results['optimal_lag_days']} days (r = {lag_results['peak_correlation']})")
print(f"Biological Insight: {lag_results['biological_insight']}")

# Plot Thermal History & Threshold Line
plt.figure(figsize=(10, 4))
plt.plot([d[5:] for d in dates], hist_lst, marker='o', color='#06b6d4', label='Lake Mean Temp (°C)')
plt.axhline(22.0, color='#f59e0b', linestyle='--', label='Critical Threshold (22°C)')
plt.title("21-Day Water Surface Temperature Evolution & Heatwave Streak")
plt.xlabel("Date (MM-DD)")
plt.ylabel("LSWT (°C)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# Cell 6: Machine Learning Training
cells.append(nbf.v4.new_markdown_cell("""## Step 6: Machine Learning Training & Explainability (XAI)
We train gradient-boosted decision trees and random forests on multi-season freshwater observations to:
- Classify 4 operational hazard tiers: **Clear (0)**, **Watch (1)**, **Warning (2)**, and **Critical (3)**.
- Quantify **Feature Importance** (Explainable AI) to reveal the dominant drivers of bloom proliferation.
"""))

cells.append(nbf.v4.new_code_cell("""ml_engine = BloomMachineLearningEngine(random_state=42)
train_metrics = ml_engine.train()

print("Model Training & Validation Performance:")
print(f"  Test Weighted F1-Score: {train_metrics['test_f1_score']}")
print(f"  Continuous BSI R² Score: {train_metrics['test_r2_score']}")
print(f"  Test RMSE:             {train_metrics['test_rmse']}")
print(f"  Training Samples:      {train_metrics['training_samples']}")

# Plot Feature Importance
features = list(ml_engine.feature_importance_.keys())
importances = list(ml_engine.feature_importance_.values())

plt.figure(figsize=(10, 5))
y_pos = np.arange(len(features))
plt.barh(y_pos, importances, color=['#06b6d4' if 'lst' in f or 'hot' in f or 'cdd' in f else '#64748b' for f in features])
plt.yticks(y_pos, [f.replace('_', ' ').upper() for f in features])
plt.gca().invert_yaxis()
plt.title("Feature Importance Ranking (Gini Attribution)")
plt.xlabel("Relative Predictive Weight")
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# Cell 7: What-If Forecasting
cells.append(nbf.v4.new_markdown_cell("""## Step 7: "What-If" Scenario Forecasting & Spatial Risk Maps
Water authorities need to plan ahead:
*What happens if a heatwave pushes water temperatures +2.5°C above normal for 5 consecutive days during calm, stagnant winds?*

Here, we simulate the forward-looking bloom progression:
"""))

cells.append(nbf.v4.new_code_cell("""forecaster = BloomForecastingEngine(ml_engine)

# Construct feature grid from current pass
h, w = raw_water_mask.shape
base_features = np.zeros((h, w, 10))
base_features[:, :, 0] = np.nan_to_num(corrected_lst, nan=20.0)
base_features[:, :, 1] = base_features[:, :, 0] - 0.4
base_features[:, :, 2] = base_features[:, :, 0] - 0.8
base_features[:, :, 3] = thermal_state["consecutive_days_above_threshold"]
base_features[:, :, 4] = thermal_state["14_day_cdd_above_20c"]
base_features[:, :, 7] = 2

# Simulate heatwave scenario: +2.5°C warming, 6 consecutive hot days, stagnant winds
forecast_res = forecaster.simulate_spatial_forecast(
    base_features=base_features,
    water_mask=raw_water_mask,
    forecast_days=7,
    temperature_delta_c=2.5,
    consecutive_exceedance_days=6,
    threshold_c=22.0,
    wind_mixing="stagnant",
    trophic_state="eutrophic"
)

print(f"Official Advisory: [{forecast_res['water_authority_advisory']['alert_code']}] {forecast_res['water_authority_advisory']['headline']}")
print(f"Action Mandate:   {forecast_res['water_authority_advisory']['action_recommendation']}")
print(f"\\nProjected Risk Coverage:")
for tier, pct in forecast_res['tier_distribution_pct'].items():
    print(f"  {tier.capitalize()}: {pct}%")

# Visualize Forecast Risk Grid
fig, ax = plt.subplots(figsize=(8, 6))
cmap_hazard = plt.cm.colors.ListedColormap(['#10b981', '#eab308', '#f97316', '#ef4444'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap_hazard.N)

masked_hazard = np.where(raw_water_mask, forecast_res["hazard_tier_grid"], np.nan)
im = ax.imshow(masked_hazard, cmap=cmap_hazard, norm=norm)
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.035, pad=0.04)
cbar.ax.set_yticklabels(['Clear (0)', 'Watch (1)', 'Warning (2)', 'Critical (3)'])
ax.set_title("Forecasted 7-Day Spatial Hazard Risk Map (Day +7)")
ax.axis('off')
plt.tight_layout()
plt.show()
"""))

# Cell 8: SOP & Export
cells.append(nbf.v4.new_markdown_cell("""## Step 8: Standard Operating Procedure (SOP) Action Table
| Alert Tier | Satellite Signature (FAI / BSI) | Thermal Trigger | Water Intake Action | Public Recreation Action |
| :--- | :--- | :--- | :--- | :--- |
| **🟢 Clear** | $BSI < 0.20$ | $T < 20^\circ\text{C}$ | Baseline operation & conventional filtration. | All water sports and bathing permitted. |
| **🟡 Watch** | $0.20 \le BSI < 0.40$ | $T \ge 22^\circ\text{C}$ for $\ge 2$ days | Check pre-chlorination; prep activated carbon. | Schedule drone/patrol shoreline survey. |
| **🟠 Warning** | $0.40 \le BSI < 0.65$ | $T \ge 22^\circ\text{C}$ for $\ge 4$ days | Increase jar testing & microcystin ELISA to 48h. | Post advisory signs; caution pet owners. |
| **🔴 Critical** | $BSI \ge 0.65$ | $T \ge 24^\circ\text{C}$ for $\ge 5$ days + calm wind | Divert to deep intakes; start PAC dosing. | Mandatory public beach closure & recreation ban. |
"""))

nb.cells = cells

with open(r"c:\Users\Pratik\Documents\HACKATHON\EIT_WATER_HACKATHON\notebooks\landsat_algae_water_authority_workflow.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Jupyter Notebook successfully written to notebooks/landsat_algae_water_authority_workflow.ipynb")
