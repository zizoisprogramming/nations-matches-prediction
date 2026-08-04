from pathlib import Path


REQ_TO_TRAIN = 50
DATA_PATH = Path(__file__).parent.parent / "data" / "staging" / "scraped.csv"
NEW_DATA_PATH = Path(__file__).parent.parent / "data" / "staging" / "new_matches.csv"
TEST_DATA_PATH = Path(__file__).parent.parent / "data" / "test" / "test.csv"