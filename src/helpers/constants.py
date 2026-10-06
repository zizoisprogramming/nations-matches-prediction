from pathlib import Path


# Retrain only when at least this many usable new matches were scraped; TRAIN_RATIO of
# them go to training and the rest to the test set.
MIN_NEW_MATCHES = 10
TRAIN_RATIO = 0.8
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "staging" / "scraped.csv"
NEW_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "staging" / "new_matches.csv"
TEST_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "test" / "test.csv"
# Where team form stats (last N games: ratings, shots, goals) come from for matches
# without cached Sofascore stats:
#   "auto"        live Sofascore first; anything it can't get (CAPTCHA, block, missing
#                 stats) falls back to the API-Football / seeded history
#   "apifootball" only the daily-collected API-Football history (src/collect_apifootball.py)
#   "sofascore"   only live Sofascore (the original behaviour)
FORM_SOURCE = "auto"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
APIFOOTBALL_MATCHES_PATH = CACHE_DIR / "apifootball_matches.json"
FORM_SEED_PATH = CACHE_DIR / "form_seed.json"
# Drop a team-date if its form games skip more than this many newer games the team is
# known to have played with stats (seeded history can be older than the team's latest games).
MAX_MISSING_FORM_GAMES = 1
