import pandas as pd
import argparse
from sklearn.metrics import classification_report
from src.inference import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir")

    args = parser.parse_args()
    save_dir = args.save_dir

    og_test = "/".join(save_dir.split("/")[:-1])
    df = pd.read_csv(og_test + "/test.csv")
    y_true = df['result']

    y_pred = predict(save_dir + "/scaled.csv")
    print(classification_report(y_true, y_pred))