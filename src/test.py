import pandas as pd
import argparse
from pathlib import Path

from sklearn.metrics import classification_report
from src.inference import predict


BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir")
    parser.add_argument("date")

    args = parser.parse_args()
    save_dir = args.save_dir
    date = args.date

    og_test = "/".join(save_dir.split("/")[:-1])
    df = pd.read_csv(og_test + "/test.csv")
    y_true = df['result']

    if date is not None:
        MODEL_PATH = BASE_DIR / "models" / f"{date}_best_model.pkl"
    y_pred = predict(save_dir + "/scaled.csv")
    print(classification_report(y_true, y_pred))