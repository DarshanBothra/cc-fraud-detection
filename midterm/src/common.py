"""Shared paths and helpers for the CSE558 midterm analysis."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
MIDTERM = ROOT / "midterm"
RESULTS = MIDTERM / "results"
FIGURES = MIDTERM / "figures"
PREPARED = RESULTS / "prepared.parquet"

SEED = 42
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}


def _to_builtin(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def save_json(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
