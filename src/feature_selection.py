import pandas as pd


SELECTED_FEATURES = [
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
    "away_temperature_avg",
    "day"
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
        