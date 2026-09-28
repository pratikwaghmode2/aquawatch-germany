"""
Core Water Indices Module for Freshwater Earth Observation
Supports Landsat 8/9 OLI and TIRS sensors.

Implements standard peer-reviewed indices for:
1. Water surface delineation: MNDWI (Xu 2006), AWEI (Feyisa et al. 2014)
2. Algal bloom & cyanobacteria detection: FAI (Hu 2009), NDCI (Mishra & Mishra 2012), SABI (Alawadi 2010)
3. Water surface temperature: Landsat Level-2 LST conversion to Celsius
4. Bloom severity index and tiered hazard classification
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union


def calculate_mndwi(green: np.ndarray, swir1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Modified Normalized Difference Water Index (MNDWI, Xu 2006).
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    
    Landsat 8/9: Green = Band 3, SWIR1 = Band 6
    Effective at separating water bodies from soil, vegetation, and urban infrastructure.
    """
    numerator = green.astype(np.float64) - swir1.astype(np.float64)
    denominator = green.astype(np.float64) + swir1.astype(np.float64) + eps
    mndwi = numerator / denominator
    return np.clip(mndwi, -1.0, 1.0)


def calculate_awei(green: np.ndarray, swir1: np.ndarray, 
                   nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Automated Water Extraction Index (AWEI_nsh, Feyisa et al. 2014)
    AWEI = 4 * (Green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)
    
    Landsat 8/9: Green = B3, SWIR1 = B6, NIR = B5, SWIR2 = B7
    Optimized for high-accuracy water boundary extraction, suppressing shadows
    and edge-induced classification errors.
    """
    return (4.0 * (green.astype(np.float64) - swir1.astype(np.float64)) - 
            (0.25 * nir.astype(np.float64) + 2.75 * swir2.astype(np.float64)))


def extract_water_mask(
    bands: Dict[str, np.ndarray], 
    method: str = "mndwi", 
    threshold: float = 0.0
) -> np.ndarray:
    """
    Extract binary water mask (True = Water, False = Land/Cloud/Invalid).
    
    Args:
        bands: Dictionary with band names ('green', 'swir1', 'nir', 'swir2', etc.)
        method: 'mndwi', 'awei', or 'combined'
        threshold: Water cutoff threshold (default 0.0)
    """
    if method == "mndwi":
        mndwi = calculate_mndwi(bands["green"], bands["swir1"])
        mask = mndwi > threshold
    elif method == "awei":
        awei = calculate_awei(bands["green"], bands["swir1"], bands["nir"], bands["swir2"])
        mask = awei > threshold
    elif method == "combined":
        mndwi = calculate_mndwi(bands["green"], bands["swir1"])
        awei = calculate_awei(bands["green"], bands["swir1"], bands["nir"], bands["swir2"])
        mask = (mndwi > threshold) & (awei > 0.0)
    else:
        raise ValueError(f"Unknown water extraction method: {method}")
        
    return mask.astype(bool)


def calculate_fai(
    nir: np.ndarray, 
    red: np.ndarray, 
    swir1: np.ndarray,
    lambda_nir: float = 865.0, 
    lambda_red: float = 655.0, 
    lambda_swir1: float = 1610.0
) -> np.ndarray:
    """
    Floating Algae Index (FAI, Hu 2009).
    FAI = R_nir - [R_red + (R_swir1 - R_red) * ((lambda_nir - lambda_red) / (lambda_swir1 - lambda_red))]
    
    Designed specifically for detecting floating algal scums (Microcystis, etc.).
    Linear baseline subtraction across Red, NIR, and SWIR effectively minimizes
    atmospheric scattering, aerosol variation, and thin cloud contamination.
    """
    weight = (lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)
    baseline = red.astype(np.float64) + (swir1.astype(np.float64) - red.astype(np.float64)) * weight
    fai = nir.astype(np.float64) - baseline
    return fai


def calculate_ndci(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalized Difference Chlorophyll Index (NDCI, Mishra & Mishra 2012).
    NDCI = (NIR - Red) / (NIR + Red)
    
    Adapted for Landsat 8/9 (Band 5 NIR and Band 4 Red).
    Directly correlates with chlorophyll-a concentration in turbid freshwater lakes.
    """
    num = nir.astype(np.float64) - red.astype(np.float64)
    den = nir.astype(np.float64) + red.astype(np.float64) + eps
    ndci = num / den
    return np.clip(ndci, -1.0, 1.0)


def calculate_sabi(
    nir: np.ndarray, 
    red: np.ndarray, 
    blue: np.ndarray, 
    green: np.ndarray, 
    eps: float = 1e-6
) -> np.ndarray:
    """
    Surface Algal Bloom Index (SABI, Alawadi 2010).
    SABI = (NIR - Red) / (Blue + Green)
    
    Suppresses bottom reflectance in shallow freshwater zones and isolates surface biomass.
    """
    num = nir.astype(np.float64) - red.astype(np.float64)
    den = blue.astype(np.float64) + green.astype(np.float64) + eps
    return num / den


def calculate_lst_celsius(
    band10: np.ndarray, 
    scale_factor: float = 0.00341802, 
    offset: float = 149.0
) -> np.ndarray:
    """
    Convert Landsat Collection 2 Level-2 Surface Temperature (ST_B10) to Celsius.
    Standard USGS formula:
    ST (Kelvin) = Digital_Number * scale_factor + offset
    ST (Celsius) = ST (Kelvin) - 273.15
    """
    b10_float = band10.astype(np.float64)
    # Check if data is already in Kelvin range (e.g. ~270 - 320) or raw DN
    if np.nanmean(b10_float[b10_float > 0]) > 200.0 and np.nanmean(b10_float[b10_float > 0]) < 350.0:
        lst_c = b10_float - 273.15
    elif np.nanmean(b10_float[b10_float > 0]) > 1000.0:
        lst_k = b10_float * scale_factor + offset
        lst_c = lst_k - 273.15
    else:
        # Already in Celsius
        lst_c = b10_float

    # Mask unphysical freshwater temperatures (-5C to 45C)
    lst_c = np.where((lst_c >= -5.0) & (lst_c <= 45.0), lst_c, np.nan)
    return lst_c


def compute_comprehensive_water_metrics(
    bands: Dict[str, np.ndarray],
    water_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Process full multi-spectral Landsat band stack into all core water indices.
    
    Expects bands: 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'
    """
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
    swir1 = bands["swir1"]
    swir2 = bands.get("swir2", swir1)
    blue = bands.get("blue", green * 0.8)
    thermal = bands.get("thermal", bands.get("b10", None))

    mndwi = calculate_mndwi(green, swir1)
    awei = calculate_awei(green, swir1, nir, swir2)
    
    if water_mask is None:
        water_mask = (mndwi > 0.0) | (awei > 0.0)
        
    fai = calculate_fai(nir, red, swir1)
    ndci = calculate_ndci(nir, red)
    sabi = calculate_sabi(nir, red, blue, green)
    
    lst_c = None
    if thermal is not None:
        lst_c = calculate_lst_celsius(thermal)
        
    # Mask out non-water pixels with NaN for bloom indices
    fai_water = np.where(water_mask, fai, np.nan)
    ndci_water = np.where(water_mask, ndci, np.nan)
    sabi_water = np.where(water_mask, sabi, np.nan)
    lst_water = np.where(water_mask, lst_c, np.nan) if lst_c is not None else None

    # Composite Normalized Bloom Severity Index (BSI, range [0, 1])
    # Normalized using empirical FAI and NDCI boundaries
    fai_norm = np.clip((fai_water - (-0.02)) / (0.15 - (-0.02)), 0.0, 1.0)
    ndci_norm = np.clip((ndci_water - (-0.1)) / (0.35 - (-0.1)), 0.0, 1.0)
    bsi = 0.6 * fai_norm + 0.4 * ndci_norm

    return {
        "mndwi": mndwi,
        "awei": awei,
        "water_mask": water_mask,
        "fai": fai_water,
        "ndci": ndci_water,
        "sabi": sabi_water,
        "lst_celsius": lst_water,
        "bloom_severity_index": bsi
    }


def classify_bloom_hazard_tier(
    bsi: np.ndarray,
    lst_c: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Classify water pixels into 4 standard operational hazard tiers:
    0: Low / Clear Water (BSI < 0.20)
    1: Watch / Early Proliferation (0.20 <= BSI < 0.40)
    2: Warning / Moderate Bloom (0.40 <= BSI < 0.65)
    3: Critical / Severe Scum Bloom (BSI >= 0.65)
    
    Returns:
        classified_grid: Integer array with values 0, 1, 2, 3 (-1 for non-water)
        tier_percentages: Area breakdown of each tier across the water body
    """
    valid_water = ~np.isnan(bsi)
    total_water_pixels = np.sum(valid_water)
    
    classified = np.full(bsi.shape, -1, dtype=np.int8)
    
    if total_water_pixels == 0:
        return classified, {"clear": 0.0, "watch": 0.0, "warning": 0.0, "critical": 0.0}
        
    t0 = valid_water & (bsi < 0.20)
    t1 = valid_water & (bsi >= 0.20) & (bsi < 0.40)
    t2 = valid_water & (bsi >= 0.40) & (bsi < 0.65)
    t3 = valid_water & (bsi >= 0.65)
    
    classified[t0] = 0
    classified[t1] = 1
    classified[t2] = 2
    classified[t3] = 3
    
    percentages = {
        "clear": float(np.sum(t0) / total_water_pixels * 100.0),
        "watch": float(np.sum(t1) / total_water_pixels * 100.0),
        "warning": float(np.sum(t2) / total_water_pixels * 100.0),
        "critical": float(np.sum(t3) / total_water_pixels * 100.0),
    }
    
    return classified, percentages
