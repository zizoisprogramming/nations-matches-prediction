from pathlib import Path


# Retrain only when at least this many usable new matches were scraped; TRAIN_RATIO of
# them go to training and the rest to the test set.
MIN_NEW_MATCHES = 10
TRAIN_RATIO = 0.8
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "staging" / "scraped.csv"
NEW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "staging" / "new_matches.csv"
TEST_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "test" / "test.csv"
# Where team form stats (last N games: ratings, shots, goals) come from for matches
# without cached Sofascore stats: "apifootball" (daily-collected, see
# src/collect_apifootball.py) or "sofascore" (live scraping).
FORM_SOURCE = "apifootball"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
APIFOOTBALL_MATCHES_PATH = CACHE_DIR / "apifootball_matches.json"
