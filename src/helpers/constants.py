from pathlib import Path


# Retrain only when at least this many usable new matches were scraped; TRAIN_RATIO of
# them go to training and the rest to the test set.
MIN_NEW_MATCHES = 10
TRAIN_RATIO = 0.8
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "staging" / "scraped.csv"
NEW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "staging" / "new_matches.csv"
TEST_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "test" / "test.csv"