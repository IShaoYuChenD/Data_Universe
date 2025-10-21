from typing import Tuple, Dict, Optional
import json
import numpy as np
import pandas as pd

def sample_lat_lon_within_bbox(n: int, bbox: Tuple[float, float, float, float], seed: Optional[int] = None) -> pd.DataFrame:
    """
    Uniformly sample n latitude/longitude points within the bounding box.

    bbox: (min_lat, min_lon, max_lat, max_lon)
    Returns DataFrame with columns ['latitude', 'longitude'] of length n.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    min_lat, min_lon, max_lat, max_lon = bbox
    if not (min_lat < max_lat and min_lon < max_lon):
        raise ValueError("bbox must be (min_lat, min_lon, max_lat, max_lon) with min < max")

    lats = rng.uniform(min_lat, max_lat, size=n)
    lons = rng.uniform(min_lon, max_lon, size=n)
    return pd.DataFrame({"latitude": lats, "longitude": lons})

def sample_replacement_values(n: int, distribution_params: Dict = None, seed: Optional[int] = None) -> pd.Series:
    """
    Sample replacement values for properties.

    distribution_params example (defaults used if None):
      {
        "type": "lognormal",
        "mu": 12.0,      # log-scale mean
        "sigma": 0.8     # log-scale std
      }

    Returns a pandas Series of positive floats (same length n).
    Units: currency units (user must record whether USD/EUR).
    """
    if distribution_params is None:
        distribution_params = {"type": "lognormal", "mu": 12.0, "sigma": 0.8}

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dtype = distribution_params.get("type", "lognormal").lower()
    if dtype == "lognormal":
        mu = float(distribution_params.get("mu", 12.0))
        sigma = float(distribution_params.get("sigma", 0.8))
        samples = rng.lognormal(mean=mu, sigma=sigma, size=n)
    elif dtype == "uniform":
        low = float(distribution_params.get("low", 50000))
        high = float(distribution_params.get("high", 500000))
        samples = rng.uniform(low, high, size=n)
    else:
        raise ValueError(f"Unknown distribution type: {dtype}")

    # enforce positive and reasonable lower bound
    samples = np.clip(samples, 1000.0, None)
    return pd.Series(samples)

def generate_synthetic_exposure(
    n: int = 1000,
    region_bbox: tuple = (34.0, -121.5, 38.5, -118.7),
    building_type_probs: Optional[dict] = None,
    value_distribution: Optional[dict] = None,
    year_built_range: tuple = (1900, 2020),
    deductible_pct: float = 0.01,
    occupancy: str = "residential",
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a synthetic exposure portfolio.

    Returns DataFrame with columns:
      id, latitude, longitude, building_type, replacement_value,
      year_built, deductible, deductible_pct, occupancy
    """
    # defaults
    if building_type_probs is None:
        building_type_probs = {"wood": 0.75, "masonry": 0.20, "rc": 0.05}
        # A dictionary defining the possible construction types and their probabilities (weights) in the portfolio.
    if value_distribution is None:
        value_distribution = {"type": "lognormal", "mu": 12.0, "sigma": 0.8}

    # RNG
    rng = np.random.default_rng(seed)

    # sample coords and values
    coords_df = sample_lat_lon_within_bbox(n=n, bbox=region_bbox, seed=seed)
    values = sample_replacement_values(n=n, distribution_params=value_distribution, seed=seed)

    # sample building types
    types = list(building_type_probs.keys())
    probs = np.array(list(building_type_probs.values()), dtype=float)
    probs = probs / probs.sum()
    building_types = rng.choice(types, size=n, p=probs)
    # randomly selects one of the building types for each of the $n$ properties, with the selection weighted by the defined probabilities.

    # year_built
    min_y, max_y = year_built_range
    years = rng.integers(min_y, max_y + 1, size=n) # uniformly sample

    # compute deductibles as fraction of replacement value
    deductibles = (values * float(deductible_pct)).round(2)
    # Deductible here refers to the amount of loss that the property owner (the insured party) must bear 
    # before the insurance company starts paying for the rest of the damage.

    # assemble dataframe
    df = pd.DataFrame({
        "id": [f"PROP_{i+1:06d}" for i in range(n)],
        "latitude": coords_df["latitude"].values,
        "longitude": coords_df["longitude"].values,
        "building_type": building_types,
        "replacement_value": values.values,
        "year_built": years,
        "deductible": deductibles,
        "deductible_pct": float(deductible_pct),
        "occupancy": occupancy
    })

    return df

def save_metadata(metadata: dict, path: str):
    """Save a small metadata JSON to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
