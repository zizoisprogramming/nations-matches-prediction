import joblib

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"

def predict_proba(
    path,
    model_path: Path = MODEL_PATH
) -> pd.DataFrame:

    df = pd.read_csv(path)
    X = df.copy()

    model = joblib.load(model_path)
    probs = model.predict_proba(X)  

    return pd.DataFrame(
        probs,
        columns=['prob_draw', 'prob_home_win', 'prob_away_win']
    )