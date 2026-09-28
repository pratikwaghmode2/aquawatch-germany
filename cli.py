"""
Operational Command-Line Interface (CLI) for Water Authorities
Enables automated daily batch processing, headless alert generation, and cron integration.
"""

import sys
import json
import argparse
from datetime import datetime
from core.landsat_pipeline import LandsatPipeline, BENCHMARK_BASINS
from core.water_indices import (
    compute_comprehensive_water_metrics,
    classify_bloom_hazard_tier
)
from core.resolution_adapter import ResolutionAdapter
from core.thermal_dynamics import ThermalDynamicsAnalyzer
from core.ml_models import BloomMachineLearningEngine
from core.forecasting_engine import BloomForecastingEngine


def main():
    parser = argparse.ArgumentParser(
        description="Freshwater Earth Observation Algal Bloom & Thermal Early Warning CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: list-basins
    subparsers.add_parser("list-basins", help="List pre-configured benchmark basins")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate current EO satellite pass and thermal status")
    eval_parser.add_argument("--basin", type=str, default="lake_balaton", choices=list(BENCHMARK_BASINS.keys()), help="Target basin key")
    eval_parser.add_argument("--heatwave", action="store_true", help="Simulate active heatwave conditions")
    eval_parser.add_argument("--threshold", type=float, default=22.0, help="Thermal trigger threshold in Celsius")

    # Command: forecast
    fc_parser = subparsers.add_parser("forecast", help="Run What-If scenario forecasting under thermal pre-conditions")
    fc_parser.add_argument("--basin", type=str, default="lake_balaton", choices=list(BENCHMARK_BASINS.keys()), help="Target basin key")
    fc_parser.add_argument("--threshold", type=float, default=22.0, help="Temperature threshold in Celsius")
    fc_parser.add_argument("--consecutive-days", type=int, default=5, help="Consecutive days exceeding threshold")
    fc_parser.add_argument("--delta", type=float, default=2.0, help="Simulated warming anomaly (+°C)")
    fc_parser.add_argument("--wind", type=str, default="stagnant", choices=["stagnant", "moderate", "turbulent"], help="Wind mixing regime")
    fc_parser.add_argument("--trophic", type=str, default="eutrophic", choices=["oligotrophic", "mesotrophic", "eutrophic", "hypereutrophic"], help="Trophic state")
    fc_parser.add_argument("--output", type=str, default=None, help="Save forecast summary JSON to path")

    args = parser.parse_args()

    if args.command == "list-basins":
        print("\n=== PRE-CONFIGURED FRESHWATER BENCHMARK BASINS ===")
        for key, val in BENCHMARK_BASINS.items():
            print(f"[{key}] {val['name']} ({val['country']})")
            print(f"   Scale: {val['scale_type']} | Area: {val['area_km2']} km² | Depth: {val['mean_depth_m']} m")
            print(f"   Primary Bloom Taxa: {val['primary_bloom_taxa']}")
            print(f"   Description: {val['description']}\n")

    elif args.command == "evaluate":
        pipeline = LandsatPipeline()
        adapter = ResolutionAdapter()
        analyzer = ThermalDynamicsAnalyzer()

        print(f"\n[+] Fetching Landsat 8/9 Level-2 scene for basin: {args.basin} (Heatwave={args.heatwave})...")
        scene = pipeline.generate_benchmark_basin_scene(args.basin, heatwave_condition=args.heatwave)
        meta = scene["metadata"]

        indices = compute_comprehensive_water_metrics(scene["bands"])
        scale_info = adapter.classify_system_scale(indices["water_mask"])
        pure_mask, _ = adapter.extract_pure_water_mask(indices["water_mask"])
        corrected_lst = adapter.harmonize_thermal_resolution(indices["lst_celsius"], indices["water_mask"], pure_mask)
        hazard_grid, percentages = classify_bloom_hazard_tier(indices["bloom_severity_index"], corrected_lst)

        thermal_state = analyzer.compute_lake_thermal_state(
            corrected_lst, 
            np.array(scene["historical_mean_lst"]), 
            threshold_c=args.threshold
        )

        print("\n================= WATER AUTHORITY EO STATUS REPORT =================")
        print(f" Basin:             {meta['name']} ({meta['country']})")
        print(f" Acquisition Date:  {scene['acquisition_date']} | Scene: {scene['scene_id']}")
        print(f" Scale Category:    {scale_info['scale_name'].upper()} ({scale_info['surface_area_km2']} km²)")
        print(f" Water Pixels:      {scale_info['water_pixel_count']:,} (Pure core: {int(np.sum(pure_mask)):,})")
        print("--------------------------------------------------------------------")
        print(f" Mean Surface Temp: {thermal_state['mean_water_temp_c']} °C (Max: {thermal_state['max_water_temp_c']} °C)")
        print(f" Consecutive Hot:   {thermal_state['consecutive_days_above_threshold']} days (Threshold: {args.threshold} °C)")
        print(f" 14-day Heat Load:  {thermal_state['14_day_cdd_above_20c']} °C·days (Status: {thermal_state['heatwave_status']})")
        print("--------------------------------------------------------------------")
        print(f" HAZARD TIERS:      Clear: {percentages['clear']}% | Watch: {percentages['watch']}% | Warning: {percentages['warning']}% | Critical: {percentages['critical']}%")
        
        if percentages['critical'] > 10 or (percentages['critical'] + percentages['warning']) > 35:
            print(" >> OVERALL MANDATE: [RED ALERT] Critical bloom detected! Activate municipal drinking intake diversion.")
        elif percentages['warning'] > 15:
            print(" >> OVERALL MANDATE: [ORANGE WARNING] Elevated bloom probability. Increase field sampling to 48h.")
        else:
            print(" >> OVERALL MANDATE: [CLEAR / WATCH] Normal seasonal regime. Routine monitoring.")
        print("====================================================================\n")

    elif args.command == "forecast":
        import numpy as np
        pipeline = LandsatPipeline()
        adapter = ResolutionAdapter()
        ml = BloomMachineLearningEngine()
        ml.train()
        forecaster = BloomForecastingEngine(ml)

        scene = pipeline.generate_benchmark_basin_scene(args.basin)
        indices = compute_comprehensive_water_metrics(scene["bands"])
        water_mask = indices["water_mask"]
        h, w = water_mask.shape

        # Construct feature grid
        features = np.zeros((h, w, 10))
        features[:, :, 0] = np.nan_to_num(indices["lst_celsius"], nan=22.0)
        features[:, :, 1] = features[:, :, 0]
        features[:, :, 2] = features[:, :, 0]
        features[:, :, 3] = args.consecutive_days
        features[:, :, 4] = 25.0
        features[:, :, 7] = 2

        print(f"\n[+] Running What-If Bloom Forecast for {args.basin}...")
        print(f"    Pre-conditions: Temp Threshold = {args.threshold}°C, Hot Days = {args.consecutive_days}, Warming Delta = +{args.delta}°C, Wind = {args.wind}")
        
        fc = forecaster.simulate_spatial_forecast(
            base_features=features,
            water_mask=water_mask,
            forecast_days=7,
            temperature_delta_c=args.delta,
            consecutive_exceedance_days=args.consecutive_days,
            threshold_c=args.threshold,
            wind_mixing=args.wind,
            trophic_state=args.trophic
        )

        dist = fc["tier_distribution_pct"]
        adv = fc["water_authority_advisory"]

        print("\n================== 7-DAY FORECAST RISK SUMMARY ==================")
        print(f" Projected Mean Bloom Severity (BSI): {fc['mean_projected_bsi']}")
        print(f" Projected Area Coverage:")
        print(f"   - Clear Water:     {dist['clear']}%")
        print(f"   - Watch / Advise:  {dist['watch']}%")
        print(f"   - Warning Tier:    {dist['warning']}%")
        print(f"   - Critical Scum:   {dist['critical']}%")
        print("-----------------------------------------------------------------")
        print(f" OFFICIAL ACTION ADVISORY: [{adv['alert_code']}] {adv['headline']}")
        print(f" Action Mandate: {adv['action_recommendation']}")
        print("=================================================================\n")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(fc, f, indent=2, default=str)
            print(f"[✓] Forecast saved to {args.output}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
