

import pandas as pd
import os 
from pathlib import Path
import datetime as dt
import joblib
import argparse

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from src.feature_extraction import FeatureExtraction
from src.feature_scaling import FeatureScaling
from src.feature_selection import FeatureSelection
from src.inference import predict_proba
from src.helpers.constants import NEW_DATA_PATH

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="date of the run")

    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent.parent
    save_dir = BASE_DIR / args.date

    try:
        os.makedirs(save_dir, exist_ok=True)
    except:
        raise Exception("Couldn't make dir", save_dir)

    fe = FeatureExtraction()
    path = fe.run(NEW_DATA_PATH, save_dir)

    fsc = FeatureScaling()
    path = fsc.run(path, save_dir)

    fsl = FeatureSelection()
    path = fsl.run(path, save_dir)

    df = pd.read_csv(path, index=False)
    X, y = df.drop(columns=['result']), df['result']

    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weights = dict(zip(classes, weights))
    sample_weights = y.map(class_weights)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss"
    )

    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    svc = SVC(
        probability=True,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("lr", lr),
            ("svc", svc)
        ],
        voting="soft"      
    )

    param_grid = {
        "xgb__n_estimators": [100, 300, 500],
        "xgb__learning_rate": [0.01, 0.03, 0.05],
        "xgb__max_depth": [3, 5],
        "xgb__subsample": [0.8, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0],

        "lr__C": [0.05, 0.1, 1],

        "svc__C": [0.05, 0.1, 1],
        "svc__gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        ensemble,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X, y, sample_weight=sample_weights)

    best_model = grid.best_estimator_

    print("Best Parameters:")
    print(grid.best_params_)

    print("Best CV Accuracy:")
    print(grid.best_score_)

    y_pred = best_model.predict(X)
    probs = best_model.predict_proba(X)

    print(probs.shape)
    print(probs[:5])

    print("Train Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

    joblib.dump(best_model, f"../models/{args.date}/best_model.pkl")

if __name__ == "__main__":
    main()