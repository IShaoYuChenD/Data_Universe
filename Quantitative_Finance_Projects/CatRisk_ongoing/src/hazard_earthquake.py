# src/hazard_earthquake.py
"""
Earthquake hazard utilities for CatRisk (Goal 2).

"""

from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# -------------------------
# Haversine (km) - vectorized
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (km). All inputs may be scalars or numpy arrays (broadcastable).
    """
    R_earth = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R_earth * c

# -------------------------
# Gutenberg-Richter magnitude sampler (inverse transform)
# -------------------------
def sample_magnitudes(n, b=1.0, m_min=4.5, m_max=8.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    beta = b * np.log(10)
    u = rng.random(n)
    c = np.exp(-beta * m_min) - np.exp(-beta * m_max)
    mags = -np.log(np.exp(-beta * m_min) - u * c) / beta
    return mags
# The Gutenberg-Richter law is a fundamental relationship in seismology that describes the relationship between the magnitude and total number of earthquakes in any given region and time period.
# n: The number of magnitudes (earthquakes) to simulate.
# b: A key parameter in the law that controls the slope of the distribution. A higher $b$ means more small earthquakes and fewer large ones.
# m_min, m_max: The minimum and maximum magnitudes to consider.
# -------------------------
# Simple attenuation (illustrative) -> PGA in g
# PGA = 10^(a0 + a1*M) / (R + c)^gamma
# -------------------------
def pga_from_mag_dist(M, R_km, a0=-3.0, a1=0.5, c=10.0, gamma=1.3):
    M = np.asarray(M)
    R = np.asarray(R_km)
    numer = 10 ** (a0 + a1 * M)
    denom = (R + c) ** gamma
    return numer / denom
# this estimates the Peak Ground Acceleration (PGA)—a measure of ground shaking intensity—based on the earthquake's magnitude and the distance from the earthquake source.
# -------------------------
# Catalog generator (epicenters uniform in bbox)
# -------------------------
def generate_earthquake_events(n_events: int = 100,
                               region_bounds: Optional[Dict] = None,
                               magnitude_range: Tuple[float,float] = (4.5, 7.5),
                               b: float = 1.0,
                               depth_km: float = 10.0,
                               seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate a synthetic catalog DataFrame:
      columns: event_id, lon, lat, M, depth_km
    region_bounds: {'lon_min','lon_max','lat_min','lat_max'} or None (defaults to a sample bbox)
    """
    rng = np.random.default_rng(seed)
    if region_bounds is None:
        # default bbox near Central Europe — change as needed
        region_bounds = {'lon_min': 5.0, 'lon_max': 15.0, 'lat_min': 45.0, 'lat_max': 55.0}

    lons = rng.uniform(region_bounds['lon_min'], region_bounds['lon_max'], n_events)
    lats = rng.uniform(region_bounds['lat_min'], region_bounds['lat_max'], n_events)
    # sample magnitudes via truncated Gutenberg-Richter for realism
    mags = sample_magnitudes(n_events, b=b, m_min=magnitude_range[0], m_max=magnitude_range[1], rng=rng)
    depths = rng.normal(loc=depth_km, scale=5.0, size=n_events).clip(1.0, 100.0)

    df = pd.DataFrame({
        'event_id': np.arange(1, n_events+1, dtype=int),
        'lon': lons,
        'lat': lats,
        'M': mags,
        'depth_km': depths
    })
    return df
# The function generates a synthetic earthquake catalog, which is a simulated dataset of earthquake events, complete with locations, magnitudes, and depths.
# -------------------------
# Compute intensities at exposures for a catalog (vectorized per event)
# Returns DataFrame with event-exposure rows
# -------------------------
def compute_intensity_for_catalog(catalog_df: pd.DataFrame,
                                  exposures_df: pd.DataFrame,
                                  attenuation_fn=pga_from_mag_dist,
                                  exposure_id_col: str = 'id',
                                  exposure_lat_col: str = 'lat',
                                  exposure_lon_col: str = 'lon',
                                  drop_zeros: bool = True) -> pd.DataFrame:
    """
    For each event in catalog_df compute PGA at every exposure point.
    catalog_df columns: 'event_id','lon','lat','M','depth_km'
    exposures_df must have columns identified by exposure_*_col
    Returns concatenated DataFrame with columns:
      event_id, exposure_id, distance_km, M, pga_g
    Note: This implementation is straightforward; for large data use chunking / KD-tree.
    """
    exp_ids = exposures_df[exposure_id_col].to_numpy()
    exp_lats = exposures_df[exposure_lat_col].to_numpy()
    exp_lons = exposures_df[exposure_lon_col].to_numpy()

    rows = []
    for _, ev in catalog_df.iterrows():
        # horizontal epicentral distance (km)
        R_epi = haversine_km(ev['lat'], ev['lon'], exp_lats, exp_lons)
        # hypocentral distance
        R_hyp = np.sqrt(R_epi**2 + (ev['depth_km'])**2)
        pga = attenuation_fn(ev['M'], R_hyp)
        df_ev = pd.DataFrame({
            'event_id': int(ev['event_id']),
            'exposure_id': exp_ids,
            'distance_km': R_hyp,
            'M': float(ev['M']),
            'pga_g': pga
        })
        rows.append(df_ev)
    if not rows:
        return pd.DataFrame(columns=['event_id','exposure_id','distance_km','M','pga_g'])
    out = pd.concat(rows, ignore_index=True)
    if drop_zeros:
        out = out[out['pga_g'] > 1e-12].reset_index(drop=True)
    return out
# It combines all the previous functions and concepts to calculate the ground shaking intensity (PGA) from an entire catalog of simulated earthquakes at every specific location of interest
# -------------------------
# High-level wrappers
# -------------------------
def simulate_catalog_and_intensities(exposures_df: pd.DataFrame,
                                     n_events=100,
                                     region_bounds: Optional[Dict]=None,
                                     magnitude_range=(4.5,7.5),
                                     b=1.0,
                                     depth_km=10.0,
                                     seed: Optional[int]=None,
                                     lambda_per_year: Optional[float]=None):
    """
    Convenience wrapper: generate catalog and compute intensities for given exposures.
    Returns (catalog_df, intensities_df)
    """
    cat = generate_earthquake_events(n_events=n_events,
                                     region_bounds=region_bounds,
                                     magnitude_range=magnitude_range,
                                     b=b,
                                     depth_km=depth_km,
                                     seed=seed)
    intens = compute_intensity_for_catalog(cat, exposures_df)
    return cat, intens

# -------------------------
# I/O helpers
# -------------------------
def save_df(df: pd.DataFrame, path: str, index=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)

def load_exposures(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
