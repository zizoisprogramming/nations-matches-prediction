import joblib

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

def predict_proba(
    path
) -> pd.DataFrame:

    df = pd.read_csv(path)
    X = df.copy()

    model = joblib.load(MODEL_PATH)
    probs = model.predict_proba(X)  

    return pd.DataFrame(
        probs,
        columns=['prob_draw', 'prob_home_win', 'prob_away_win']
    )

def predict(
    path
) -> pd.DataFrame:

    df = pd.read_csv(path)
    X = df.copy()[:5]

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X)  

    return pred