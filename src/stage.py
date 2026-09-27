import numpy as np
import pandas as pd

from src.feature_extraction import FeatureExtraction
from src.helpers.constants import MIN_NEW_MATCHES, TRAIN_RATIO, DATA_PATH, NEW_DATA_PATH, TEST_DATA_PATH


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
    Turns the n matches buffered in scraped.csv into new_matches.csv for auto-train,
    once at least MIN_NEW_MATCHES of them have usable form stats:
      - TRAIN_RATIO of the n new matches go to training,
      - the same number as the rest, (1 - TRAIN_RATIO) * n, is taken from the start of
        test.csv into training too (or all of test.csv if it has fewer rows),
      - the rest of the new matches replace them in test.csv.
    So training grows by n and test.csv keeps its size. scraped.csv is emptied
    afterwards so the same rows are never staged twice.
    """
    try:
        df = pd.read_csv(DATA_PATH).drop_duplicates()
    except FileNotFoundError:
        df = pd.DataFrame()

    if len(df) < MIN_NEW_MATCHES:
        print(f"Not enough new matches to train ({len(df)}/{MIN_NEW_MATCHES}).")
        return

    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    label_data(df)

    # Compute every buffered match's form stats (fresh from Sofascore) before the
    # events cache is trimmed; the rows sent to test.csv aren't extracted until later.
    df = FeatureExtraction().warm_form_cache(df)

    n = len(df)
    if n < MIN_NEW_MATCHES:
        # Keep the usable ones buffered so they count towards next week's batch.
        df.drop(columns=["result"]).to_csv(DATA_PATH, index=False)
        print(f"Only {n} new matches have usable form stats ({n}/{MIN_NEW_MATCHES}); "
              f"kept them in {DATA_PATH.name} for the next run.")
        return

    test_df = pd.read_csv(TEST_DATA_PATH).drop_duplicates()

    n_to_test = round((1 - TRAIN_RATIO) * n)
    n_train_new = n - n_to_test
    n_from_test = min(n_to_test, len(test_df))

    new_data = pd.concat([df.iloc[:n_train_new], test_df.iloc[:n_from_test]], ignore_index=True)
    new_data.to_csv(NEW_DATA_PATH, index=False)

    updated_test_df = pd.concat([test_df.iloc[n_from_test:], df.iloc[n_train_new:]], ignore_index=True)
    updated_test_df.to_csv(TEST_DATA_PATH, index=False)

    df.iloc[0:0].drop(columns=["result"]).to_csv(DATA_PATH, index=False)

    print(f"Training on {n_train_new} new + {n_from_test} from test; moved {n_to_test} new to test "
          f"(test size {len(test_df)} -> {len(updated_test_df)}).")


if __name__ == "__main__":
    main()
