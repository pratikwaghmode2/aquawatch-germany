"""
Dynamic Bloom Forecasting & Risk Mapping Engine
Simulates multi-day bloom development and produces forward-looking risk maps under:
- Consecutive hot day exceedance scenarios
- Water temperature warming anomalies (+1°C to +4°C)
- Atmospheric stagnation / wind-mixing states
- Eutrophication baseline pre-conditions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from core.ml_models import BloomMachineLearningEngine


class BloomForecastingEngine:
    """
    Predictive engine simulating future bloom proliferation and producing actionable risk maps.
    """

    def __init__(self, ml_engine: Optional[BloomMachineLearningEngine] = None):
        if ml_engine is None:
            self.ml_engine = BloomMachineLearningEngine()
            self.ml_engine.train()
        else:
            self.ml_engine = ml_engine

    def simulate_spatial_forecast(
        self,
        base_features: np.ndarray,
        water_mask: np.ndarray,
        forecast_days: int = 7,
        temperature_delta_c: float = 2.0,
        consecutive_exceedance_days: int = 5,
        threshold_c: float = 22.0,
        wind_mixing: str = "stagnant",
        trophic_state: str = "eutrophic"
    ) -> Dict[str, Any]:
        """
        Simulate future spatial bloom map under specified climate and thermal pre-conditions.
        
        Args:
            base_features: [H, W, 10] array of current feature states
            water_mask: [H, W] boolean mask of water pixels
            forecast_days: Simulation horizon (e.g. 1, 3, 7, 14 days)
            temperature_delta_c: Simulated temperature increase (+°C)
            consecutive_exceedance_days: Consecutive days exceeding threshold
            threshold_c: Thermal trigger temperature (°C)
            wind_mixing: 'stagnant' (promotes scum), 'moderate', or 'turbulent' (disperses scum)
            trophic_state: 'oligotrophic', 'mesotrophic', 'eutrophic', 'hypereutrophic'
        """
        h, w, c = base_features.shape
        future_features = base_features.copy()

        # Wind mixing damping or acceleration factor
        wind_multipliers = {
            "stagnant": 1.25,   # Calm weather allows buoyant cyanobacteria to aggregate into thick surface mats
            "moderate": 1.00,
            "turbulent": 0.65   # High wind breaks thermocline and disperses surface scum into water column
        }
        wind_factor = wind_multipliers.get(wind_mixing.lower(), 1.0)

        # Trophic state baseline nutrient factor
        trophic_offsets = {
            "oligotrophic": -0.15,
            "mesotrophic": -0.05,
            "eutrophic": 0.10,
            "hypereutrophic": 0.25
        }
        trophic_offset = trophic_offsets.get(trophic_state.lower(), 0.05)

        # Update feature channels according to the what-if pre-conditions:
        # Channel 0: lst_current -> add delta
        future_features[:, :, 0] += temperature_delta_c
        
        # Channel 1: lst_3d_mean -> adjusts towards future temp
        future_features[:, :, 1] += temperature_delta_c * 0.85
        
        # Channel 2: lst_7d_mean -> adjusts towards future temp
        future_features[:, :, 2] += temperature_delta_c * 0.70
        
        # Channel 3: consecutive_hot_days -> updated to user pre-condition
        future_features[:, :, 3] = consecutive_exceedance_days
        
        # Channel 4: cdd_20 -> grows with consecutive hot days and temp delta
        incremental_cdd = max(0.0, (threshold_c + temperature_delta_c - 20.0)) * consecutive_exceedance_days
        future_features[:, :, 4] += incremental_cdd
        
        # Channel 5: thermal_anomaly
        future_features[:, :, 5] += temperature_delta_c

        # Predict future state with ML model
        predictions = self.ml_engine.predict(future_features)
        future_bsi = predictions["bloom_severity_index"]
        
        # Apply hydrodynamic & trophic adjustments to continuous BSI
        adjusted_bsi = np.where(
            water_mask,
            np.clip((future_bsi + trophic_offset) * wind_factor, 0.0, 1.0),
            np.nan
        )

        # Re-derive hazard tiers
        hazard_tiers = np.full((h, w), -1, dtype=int)
        valid = water_mask & (~np.isnan(adjusted_bsi))
        
        hazard_tiers[valid & (adjusted_bsi < 0.20)] = 0
        hazard_tiers[valid & (adjusted_bsi >= 0.20) & (adjusted_bsi < 0.40)] = 1
        hazard_tiers[valid & (adjusted_bsi >= 0.40) & (adjusted_bsi < 0.65)] = 2
        hazard_tiers[valid & (adjusted_bsi >= 0.65)] = 3

        # Calculate hazard tier distribution
        total_valid = np.sum(valid)
        if total_valid > 0:
            pct_clear = float(np.sum(hazard_tiers == 0) / total_valid * 100.0)
            pct_watch = float(np.sum(hazard_tiers == 1) / total_valid * 100.0)
            pct_warning = float(np.sum(hazard_tiers == 2) / total_valid * 100.0)
            pct_critical = float(np.sum(hazard_tiers == 3) / total_valid * 100.0)
            mean_future_bsi = float(np.mean(adjusted_bsi[valid]))
        else:
            pct_clear = pct_watch = pct_warning = pct_critical = mean_future_bsi = 0.0

        # Multi-day trajectory progression (Day 0 to forecast_days)
        trajectory = []
        days_sequence = sorted(list(set([0, 1, 3, 5, 7, 10, 14])))
        days_sequence = [d for d in days_sequence if d <= max(forecast_days, 7)]
        
        for d in days_sequence:
            progress = d / max(forecast_days, 1)
            d_crit = pct_critical * (progress ** 1.3)
            d_warn = pct_warning * progress + (1 - progress) * (pct_watch * 0.5)
            d_mean_bsi = mean_future_bsi * (0.4 + 0.6 * progress)
            trajectory.append({
                "day": d,
                "projected_mean_bsi": round(d_mean_bsi, 3),
                "critical_risk_coverage_pct": round(d_crit, 1),
                "warning_risk_coverage_pct": round(d_warn, 1)
            })

        # Water Authority Advisory determination
        if pct_critical > 15.0 or (pct_critical + pct_warning) > 40.0:
            authority_alert_code = "RED_ALERT"
            authority_headline = "CRITICAL ALGAL BLOOM EMERGENCY"
            action_recommendation = (
                "Issue public contact recreation ban. Divert municipal drinking water intakes "
                "to secondary/deep intakes. Initiate activated carbon dosing at treatment facilities. "
                "Deploy floating boom barriers around critical harbor and marina zones."
            )
        elif pct_warning > 20.0 or (pct_critical + pct_warning) > 25.0:
            authority_alert_code = "ORANGE_WARNING"
            authority_headline = "ELEVATED BLOOM WARNING"
            action_recommendation = (
                "Increase in-situ grab sampling frequency to 48-hour intervals. Post advisory signage "
                "at public beaches. Prepare aeration bubblers in stagnant bays."
            )
        elif pct_watch > 25.0:
            authority_alert_code = "YELLOW_WATCH"
            authority_headline = "BLOOM DEVELOPMENT WATCH"
            action_recommendation = (
                "Monitor next satellite pass. Track daily degree day accumulation. "
                "Inspect upwind shoreline zones for microcystin scum accumulation."
            )
        else:
            authority_alert_code = "GREEN_NORMAL"
            authority_headline = "NORMAL WATER QUALITY"
            action_recommendation = "Standard baseline routine monitoring."

        return {
            "forecast_horizon_days": forecast_days,
            "scenario_pre_conditions": {
                "temperature_delta_c": temperature_delta_c,
                "consecutive_exceedance_days": consecutive_exceedance_days,
                "threshold_c": threshold_c,
                "wind_mixing": wind_mixing,
                "trophic_state": trophic_state
            },
            "hazard_tier_grid": hazard_tiers,
            "bloom_severity_grid": adjusted_bsi,
            "tier_distribution_pct": {
                "clear": round(pct_clear, 1),
                "watch": round(pct_watch, 1),
                "warning": round(pct_warning, 1),
                "critical": round(pct_critical, 1)
            },
            "mean_projected_bsi": round(mean_future_bsi, 3),
            "trajectory": trajectory,
            "water_authority_advisory": {
                "alert_code": authority_alert_code,
                "headline": authority_headline,
                "action_recommendation": action_recommendation
            }
        }
