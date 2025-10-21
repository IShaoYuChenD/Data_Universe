# tests/test_exposure.py
import sys
import os
import pandas as pd
# ensure repo root is in Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from src.exposure import generate_synthetic_exposure, sample_lat_lon_within_bbox, sample_replacement_values

def test_generate_size_and_columns():
    df = generate_synthetic_exposure(n=50, seed=123)
    assert len(df) == 50
    expected_cols = {"id", "latitude", "longitude", "building_type", "replacement_value",
                     "year_built", "deductible", "deductible_pct", "occupancy"}
    assert set(df.columns) >= expected_cols

def test_bbox_sampling_within_bounds():
    bbox = (10.0, 20.0, 11.0, 21.0)
    coords = sample_lat_lon_within_bbox(n=100, bbox=bbox, seed=1)
    assert coords['latitude'].between(bbox[0], bbox[2]).all()
    assert coords['longitude'].between(bbox[1], bbox[3]).all()

def test_values_positive_and_reproducible():
    s1 = sample_replacement_values(n=100, distribution_params={"type":"lognormal","mu":10,"sigma":0.5}, seed=7)
    s2 = sample_replacement_values(n=100, distribution_params={"type":"lognormal","mu":10,"sigma":0.5}, seed=7)
    # All positive
    assert (s1 > 0).all()
    # reproducible with same seed (elementwise)
    assert s1.equals(s2)

def test_csv_written_by_notebook():
    # if the notebook ran and saved the file
    csv_path = os.path.join("data", "exposure_sample.csv")
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) > 0
