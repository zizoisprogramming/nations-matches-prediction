.PHONY: train test scrape

DATA_DIR := data/staging/

scrape:
	uv run python -m src.scrape "weekly"

$(DATA_DIR)/new_matches.csv:
	uv run python -m src.scrape "weekly"

$(DATA_DIR)/extracted.csv: $(DATA_DIR)/new_matches.csv
	uv run python -m src.feature_extraction

$(DATA_DIR)/selected.csv: $(DATA_DIR)/extracted.csv
	uv run python -m src.feature_selection

$(DATA_DIR)/scaled.csv: $(DATA_DIR)/selected.csv
	uv run python -m src.feature_scaling

train: $(DATA_DIR)/scaled.csv
	uv run python -m src.train ""

test:
	uv run python -m src.test