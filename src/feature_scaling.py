import joblib

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


SCALE_COLS = [
    "home_stadium_distance_km",
    "away_stadium_distance_km",
    "home_fix_1",
    "home_fix_2",
    "home_shots_1",
    "home_shots_2",
    "home_shots_against_1",
    "home_shots_against_2",
    "home_scored_1",
    "home_scored_2",
    "home_relative_shots_1",
    "home_relative_shots_2",
    "home_relative_goals_1",
    "home_relative_goals_2",
    "away_fix_1",
    "away_fix_2",
    "away_shots_1",
    "away_shots_2",
    "away_shots_against_1",
    "away_shots_against_2",
    "away_scored_2",
    "away_relative_shots_1",
    "away_relative_shots_2",
    "away_relative_goals_1",
    "away_relative_goals_2",
    "home_ranking",
    "away_ranking",
    "home_precipitation",
    "home_wind_speed",
    "away_precipitation",
    "away_wind_speed",
    "stadium_precipitation",
    "stadium_wind_speed",
    "home_stadium_temp_avg",
    "away_stadium_temp_avg",
    "home_stadium_wind_speed",
    "away_stadium_wind_speed",
    "ranking_diff",
    "stadium_temperature_avg",
    "home_temperature_avg",
    "away_temperature_avg"
]

class FeatureScaling:

    def __init__(self):
        self.scaler = joblib.load(SCALER_PATH)

    def _add_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """Cyclic encoding for day and month columns."""
            df = df.copy()
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
            return df.drop(columns=['day'], errors='ignore')

    
    def scale(self, X):
        cols_to_scale = [c for c in SCALE_COLS if c in X.columns]
        X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])


    def run(self, path, save_dir):
        
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        X = df.drop(columns=['result'], errors='ignore').copy()
        X = self._add_cyclic_features(X)
        self.scale(X)
        X.to_csv(f"{save_dir}/scaled.csv", index=False)
        return f"{save_dir}/scaled.csv"
        