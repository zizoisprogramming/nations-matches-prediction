import asyncio
import datetime as dt
import json
import math
import time
from pathlib import Path

import joblib
import nest_asyncio
import numpy as np
import pandas as pd
import requests
from playwright.async_api import async_playwright

nest_asyncio.apply()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

CACHE_DIR = BASE_DIR / "data" / "cache"
COORDS_CACHE_PATH = CACHE_DIR / "coords_cache.json"
WEATHER_CACHE_PATH = CACHE_DIR / "weather_cache.json"
RATINGS_CACHE_PATH = CACHE_DIR / "ratings_cache.json"
CAPITALS_CACHE_PATH = CACHE_DIR / "capitals_cache.json"
TEAM_IDS_PATH = CACHE_DIR / "team_ids.json"

# ── Feature lists (must match training exactly) ───────────────────────────────
SCALE_COLS = [
    'away_stadium_distance_km',
    'home_fix_1', 'home_fix_2',
    'home_shots_against_1', 'home_shots_against_2',
    'home_relative_shots_1', 'home_relative_shots_2',
    'away_fix_1', 'away_fix_2',
    'away_relative_shots_1', 'away_relative_shots_2',
    'home_ranking', 'away_ranking',
    'home_wind_speed', 'away_wind_speed',
    'stadium_wind_speed',
    'home_stadium_temp_avg', 'away_stadium_temp_avg',
    'away_stadium_wind_speed',
    'ranking_diff',
    'stadium_temperature_avg',
    'home_temperature_avg', 'away_temperature_avg',
]

CYCLIC_COLS = ['day_sin', 'day_cos', 'month_sin', 'month_cos']

SELECTED_FEATURES = [
    'away_stadium_distance_km',
 'home_fix_1',
 'home_fix_2',
 'home_shots_against_1',
 'home_shots_against_2',
 'home_relative_shots_1',
 'home_relative_shots_2',
 'away_fix_1',
 'away_fix_2',
 'away_relative_shots_1',
 'away_relative_shots_2',
 'home_ranking',
 'away_ranking',
 'home_wind_speed',
 'away_wind_speed',
 'stadium_wind_speed',
 'home_stadium_temp_avg',
 'away_stadium_temp_avg',
 'away_stadium_wind_speed',
 'ranking_diff',
 'stadium_temperature_avg',
 'home_temperature_avg',
 'away_temperature_avg']


_LIVE_FETCHED_COLS = {
    "home_wind_speed", "away_wind_speed", "stadium_wind_speed",
    "home_temperature_max", "home_temperature_min",
    "away_temperature_max", "away_temperature_min",
    "stadium_temperature_max", "stadium_temperature_min",
    "home_ranking", "away_ranking",
    "home_fix_1", "home_fix_2", "away_fix_1", "away_fix_2",
    "home_shots_against_1", "home_shots_against_2",
    "away_shots_against_1", "away_shots_against_2",
    "home_relative_shots_1", "home_relative_shots_2",
    "away_relative_shots_1", "away_relative_shots_2",
    "away_stadium_distance_km",
}


def _needs_live_fetch(df: pd.DataFrame) -> bool:
    return not _LIVE_FETCHED_COLS.issubset(df.columns)


# ── Public API ────────────────────────────────────────────────────────────────
def predict_proba(
    df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    fetch_live: bool = True,
) -> pd.DataFrame:
    """
    Run the full inference pipeline on raw match data.

    Parameters
    ----------
    df : pd.DataFrame
        Either:
          (a) minimal match info — 'home_team', 'away_team', 'city', 'country',
              'date' (YYYY-MM-DD) — in which case weather, SofaScore form/ranking,
              and geocoded distance are fetched live, or
          (b) a fully pre-engineered row (temperature min/max, ranking, wind speed,
              shots, etc.), in which case pass fetch_live=False to skip straight to
              scaling/prediction.
        'result' is ignored if present.
    model_path, scaler_path : Path
        Paths to the saved XGBoost / RobustScaler .pkl files.
    fetch_live : bool
        Whether to fetch weather/SofaScore/geocoding data for rows missing it.

    Returns
    -------
    pd.DataFrame
        One row per input match with columns:
            prob_draw, prob_home_win, prob_away_win
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    X = df.drop(columns=['result'], errors='ignore').copy()

    if fetch_live and _needs_live_fetch(X):
        required_meta = {"home_team", "away_team", "city", "country", "date"}
        missing_meta = required_meta - set(X.columns)
        if missing_meta:
            raise ValueError(
                f"Can't fetch live features — missing columns: {sorted(missing_meta)}. "
                "Either supply them, or pass a fully pre-engineered DataFrame with "
                "fetch_live=False."
            )
        X = _add_location_features(X)
        X = _add_weather_features(X)
        X = _add_sofascore_features(X)

    if 'day' not in X.columns or 'month' not in X.columns:
        if 'date' in X.columns:
            dts = pd.to_datetime(X['date'])
            X['day'] = dts.dt.day
            X['month'] = dts.dt.month

    X = _add_derived_features(X)

    if 'day' in X.columns and 'month' in X.columns:
        X = _add_cyclic_features(X)

    scaler = joblib.load(scaler_path)
    cols_to_scale = [c for c in SCALE_COLS if c in X.columns]
    X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    missing = [c for c in SELECTED_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required features: {missing}"
        )
    X = X[SELECTED_FEATURES]

    model = joblib.load(model_path)
    probs = model.predict_proba(X)  # shape (n, 3)

    return pd.DataFrame(
        probs,
        columns=['prob_draw', 'prob_home_win', 'prob_away_win'],
        index=df.index,
    )


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_csv>")
        print("CSV needs at minimum: home_team, away_team, city, country, date")
        sys.exit(1)

    raw = pd.read_csv(sys.argv[1])
    results = predict_proba(raw)
    print(results.to_string(index=False))

print("pipeline module loaded.")