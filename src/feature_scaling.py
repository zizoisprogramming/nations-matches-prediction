import joblib

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


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

class FeatureScaling:

    def __init__(self):
        self.scaler = joblib.load(SCALER_PATH)

    def scale(self, X):
        cols_to_scale = [c for c in SCALE_COLS if c in X.columns]
        X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])


    def run(self, path, save_dir):
        
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        X = df.drop(columns=['result'], errors='ignore').copy()
        self.scale(X)
        X.to_csv(f"{save_dir}/scaled.csv")
        