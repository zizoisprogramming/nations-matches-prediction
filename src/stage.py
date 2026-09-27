import numpy as np
import pandas as pd

from src.feature_extraction import FeatureExtraction
from src.helpers.constants import REQ_TO_TRAIN, DATA_PATH, NEW_DATA_PATH, TEST_DATA_PATH


def label_data(df: pd.DataFrame):
    df['result'] = np.select(
        [
            df['home_score'] > df['away_score'],
            df['home_score'] < df['away_score']
        ],
        [
            1,
            2
        ],
        default=0
    )


def main():
    """
    Turns the matches buffered in scraped.csv into new_matches.csv for auto-train,
    once there are at least REQ_TO_TRAIN of them. All buffered rows are consumed
    (70% of REQ_TO_TRAIN to training, the rest to the test set), so scraped.csv is
    emptied afterwards and the same rows are never staged twice.
    """
    try:
        df = pd.read_csv(DATA_PATH).drop_duplicates()
    except FileNotFoundError:
        df = pd.DataFrame()

    if len(df) < REQ_TO_TRAIN:
        print(f"Not enough new matches to train ({len(df)}/{REQ_TO_TRAIN}).")
        return

    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    label_data(df)

    # Compute every buffered match's form stats (fresh from Sofascore) before the
    # events cache is trimmed; the rows sent to test.csv aren't extracted until later.
    df = FeatureExtraction().warm_form_cache(df)

    test_df = pd.read_csv(TEST_DATA_PATH).drop_duplicates()

    n_from_df = int(0.7 * REQ_TO_TRAIN)
    n_from_test = int(0.3 * REQ_TO_TRAIN)

    df_new = df.iloc[:n_from_df]
    df_rest = df.iloc[n_from_df:]

    test_new = test_df.iloc[:n_from_test]
    test_rest = test_df.iloc[n_from_test:]

    new_data = pd.concat([df_new, test_new], ignore_index=True)
    new_data.to_csv(NEW_DATA_PATH, index=False)

    updated_test_df = pd.concat([test_rest, df_rest], ignore_index=True)
    updated_test_df.to_csv(TEST_DATA_PATH, index=False)

    df.iloc[0:0].drop(columns=["result"]).to_csv(DATA_PATH, index=False)

    print(f"Staged {len(new_data)} matches to {NEW_DATA_PATH.name}, "
          f"moved {len(df_rest)} to {TEST_DATA_PATH.name}.")


if __name__ == "__main__":
    main()
