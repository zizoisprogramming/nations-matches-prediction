

import pandas as pd
from pathlib import Path
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
from sklearn.base import clone


BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "scaled_train.csv"


def grid_search(X, y, sample_weights):
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

    return best_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="date of the run")
    parser.add_argument("--search", action="store_true",
                        help="re-run the full hyperparameter grid search (slow) instead of "
                             "refitting the current best model's hyperparameters")

    args = parser.parse_args()

    # The Makefile's train target has already run extraction -> selection -> scaling
    # on the staged matches into data/<date>/.
    new_df = pd.read_csv(BASE_DIR / "data" / args.date / "scaled.csv")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    df = pd.concat([train_df, new_df], ignore_index=True)
    print(f"Training on {len(train_df)} historical + {len(new_df)} new matches")

    current_model = joblib.load(MODEL_PATH)
    feature_names = list(current_model.feature_names_in_)
    X, y = df[feature_names], df['result'].astype(int)

    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weights = dict(zip(classes, weights))
    sample_weights = y.map(class_weights)

    if args.search:
        best_model = grid_search(X, y, sample_weights)
    else:
        best_model = clone(current_model)
        best_model.fit(X, y, sample_weight=sample_weights)

    y_pred = best_model.predict(X)
    probs = best_model.predict_proba(X)

    print(probs.shape)
    print(probs[:5])

    print("Train Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

    joblib.dump(best_model, MODELS_DIR / f"{args.date}_best_model.pkl")
    joblib.dump(best_model, MODEL_PATH)
    df.to_csv(TRAIN_DATA_PATH, index=False)

if __name__ == "__main__":
    main()
