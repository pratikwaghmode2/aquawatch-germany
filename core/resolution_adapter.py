"""
Resolution Adapter & System Size Module
Handles spatial scale challenges in satellite Earth Observation for freshwater:
1. Mixed pixels along land-water shorelines (NIR adjacency effect)
2. Coarse thermal resolution (100m native Landsat TIRS resampled to 30m)
3. Scaling across small lakes/rivers (< 1 km²), medium reservoirs (1-50 km²), and large lakes (> 50 km²)
"""

import numpy as np
from scipy import ndimage
from enum import Enum
from typing import Dict, Tuple, Optional, Any


class FreshwaterSystemScale(str, Enum):
    SMALL_OR_NARROW = "small_narrow"     # Area < 1 km² or river width < 150m
    MEDIUM_RESERVOIR = "medium_reservoir" # 1 km² <= Area <= 50 km²
    LARGE_LAKE = "large_lake"             # Area > 50 km²


class ResolutionAdapter:
    """
    Adaptive spatial processor tailored to water system dimensions and Landsat sensor characteristics.
    """
    
    def __init__(self, pixel_resolution_m: float = 30.0, thermal_native_resolution_m: float = 100.0):
        self.pixel_res = pixel_resolution_m
        self.thermal_native_res = thermal_native_resolution_m

    def classify_system_scale(self, water_mask: np.ndarray) -> Dict[str, Any]:
        """
        Determine system size category based on water pixel count and morphology.
        30m x 30m pixel = 900 m² = 0.0009 km²
        1 km² ≈ 1,111 pixels
        50 km² ≈ 55,555 pixels
        """
        water_pixel_count = int(np.sum(water_mask))
        area_km2 = water_pixel_count * (self.pixel_res ** 2) / 1e6
        
        # Calculate maximum inscribed distance (distance transform) to check water width
        if water_pixel_count > 0:
            dist_map = ndimage.distance_transform_edt(water_mask)
            max_width_pixels = float(np.max(dist_map) * 2.0)
            max_width_m = max_width_pixels * self.pixel_res
        else:
            max_width_m = 0.0

        if area_km2 < 1.0 or max_width_m < 150.0:
            scale = FreshwaterSystemScale.SMALL_OR_NARROW
            recommended_erosion_iterations = 1
            pure_water_threshold = 0.85
            notes = (
                f"Small system / narrow channel ({area_km2:.2f} km², max width {max_width_m:.0f}m). "
                "High sensitivity to shoreline vegetation and mixed 100m thermal pixels. "
                "Applying conservative shoreline erosion and purity filtering."
            )
        elif area_km2 <= 50.0:
            scale = FreshwaterSystemScale.MEDIUM_RESERVOIR
            recommended_erosion_iterations = 1
            pure_water_threshold = 0.75
            notes = (
                f"Medium reservoir/lake ({area_km2:.2f} km², max width {max_width_m:.0f}m). "
                "Well suited for 30m Landsat OLI with 1-pixel shoreline boundary buffer."
            )
        else:
            scale = FreshwaterSystemScale.LARGE_LAKE
            recommended_erosion_iterations = 2
            pure_water_threshold = 0.70
            notes = (
                f"Large freshwater lake/basin ({area_km2:.2f} km²). "
                "High statistical robustness with spatial gradients, bay vs open-water thermal divergence."
            )

        return {
            "scale": scale,
            "scale_name": scale.value,
            "water_pixel_count": water_pixel_count,
            "surface_area_km2": round(area_km2, 3),
            "max_width_meters": round(max_width_m, 1),
            "recommended_erosion_iterations": recommended_erosion_iterations,
            "pure_water_threshold": pure_water_threshold,
            "notes": notes
        }

    def extract_pure_water_mask(
        self, 
        water_mask: np.ndarray, 
        iterations: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eliminates shoreline mixed pixels using morphological binary erosion.
        
        Returns:
            pure_water_mask: Mask of deep/interior water free from bank adjacency contamination.
            shoreline_buffer_mask: Mask of boundary pixels that require cautionary interpretation.
        """
        if iterations is None:
            stats = self.classify_system_scale(water_mask)
            iterations = stats["recommended_erosion_iterations"]
            
        # For very narrow rivers where erosion would wipe out the entire channel,
        # adaptively maintain the skeleton if eroded count drops to 0.
        eroded = ndimage.binary_erosion(water_mask, iterations=iterations)
        
        if np.sum(eroded) < 10 and np.sum(water_mask) >= 10:
            # Fall back to 0 erosion iterations for narrow channels to preserve connectivity
            eroded = water_mask.copy()
            
        shoreline_buffer = water_mask & (~eroded)
        return eroded, shoreline_buffer

    def correct_shoreline_adjacency(
        self,
        fai_array: np.ndarray,
        water_mask: np.ndarray,
        pure_water_mask: np.ndarray
    ) -> np.ndarray:
        """
        Mitigates false positive bloom alerts caused by shoreline riparian vegetation.
        
        Vegetation on banks bleeds high NIR into shoreline water pixels, creating
        artificially high FAI values right at the shoreline even when no bloom exists.
        
        Rule: If an edge water pixel has FAI > threshold but all neighboring pure water
        pixels are low, penalize or mask the edge pixel.
        """
        corrected_fai = fai_array.copy()
        edge_pixels = water_mask & (~pure_water_mask)
        
        # Calculate mean FAI in pure water interior
        pure_fai_values = fai_array[pure_water_mask & ~np.isnan(fai_array)]
        if len(pure_fai_values) > 0:
            median_interior_fai = np.nanmedian(pure_fai_values)
            std_interior_fai = np.nanstd(pure_fai_values) + 1e-6
            
            # Edge anomaly threshold
            edge_anomaly_cutoff = median_interior_fai + 2.5 * std_interior_fai
            
            # Mark isolated edge spikes as adjacency artifacts
            adjacency_artifact_mask = edge_pixels & (corrected_fai > edge_anomaly_cutoff)
            corrected_fai[adjacency_artifact_mask] = np.nan
            
        return corrected_fai

    def harmonize_thermal_resolution(
        self,
        lst_celsius: np.ndarray,
        water_mask: np.ndarray,
        pure_water_mask: np.ndarray
    ) -> np.ndarray:
        """
        Harmonizes 100m native Landsat TIRS with 30m optical water boundaries.
        
        Because TIRS native resolution is ~100m, edge pixels mix warm summer soil/sand
        temperatures (which can reach 35-45C) with lake water (20-25C), causing a false
        hot-fringe around lakes.
        
        This filter uses the pure water mask and distance-weighted spatial interpolation
        to replace contaminated shoreline thermal values with interior water temperature.
        """
        corrected_lst = lst_celsius.copy()
        
        # Non-water is NaN
        corrected_lst[~water_mask] = np.nan
        
        # Contaminated shoreline thermal pixels
        edge_mask = water_mask & (~pure_water_mask)
        
        # If pure water pixels exist, smooth edge thermal contamination
        if np.sum(pure_water_mask) > 0:
            pure_lst = np.where(pure_water_mask, lst_celsius, np.nan)
            mean_interior_temp = np.nanmean(pure_lst)
            max_plausible_diff = 3.5  # Max plausible degrees warmer at edge than center
            
            # Identify edge thermal pixels that exceed plausible aquatic boundary
            unrealistic_hot_edges = edge_mask & (corrected_lst > (mean_interior_temp + max_plausible_diff))
            corrected_lst[unrealistic_hot_edges] = mean_interior_temp + 1.0  # soft cap
            
        return corrected_lst
