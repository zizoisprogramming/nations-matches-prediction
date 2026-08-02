import pandas as pd


SELECTED_FEATURES = [
    'away_stadium_distance_km', 'home_fix_1', 'home_fix_2',
    'home_shots_against_1', 'home_shots_against_2', 'home_relative_shots_1',
    'home_relative_shots_2', 'away_fix_1', 'away_fix_2',
    'away_relative_shots_1', 'away_relative_shots_2',
    'home_ranking', 'away_ranking',
    'home_wind_speed', 'away_wind_speed', 'stadium_wind_speed',
    'home_stadium_temp_avg', 'away_stadium_temp_avg', 'away_stadium_wind_speed',
    'ranking_diff', 'stadium_temperature_avg',
    'home_temperature_avg', 'away_temperature_avg'
]


class FeatureSelection:

    def run(self, path, save_dir):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        X = df.drop(columns=['result'], errors='ignore').copy()
        missing = [c for c in SELECTED_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required features: {missing}"
            )
        X = X[SELECTED_FEATURES]
        X.to_csv(f"{save_dir}/selected.csv", index=False)
        return f"{save_dir}/selected.csv"
        