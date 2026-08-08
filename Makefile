.PHONY: train test scrape feature_extraction feature_selection feature_scaling 

TODAY := $(shell date +%Y-%m-%d)

STAGING_DIR := data/staging/
TEST_DIR := data/test/$(TODAY)
DATA_DIR := data/$(TODAY)
MODEL_DIR := models/
MODE := test

$(STAGING_DIR)/new_matches.csv:
	@uv run python -m src.scrape "weekly"

$(DATA_DIR)/extracted.csv:
	@mkdir -p $(DATA_DIR)
	@uv run python -m src.feature_extraction $(DATA_DIR) train

$(DATA_DIR)/selected.csv: $(DATA_DIR)/extracted.csv
	@uv run python -m src.feature_selection $(DATA_DIR)

$(DATA_DIR)/scaled.csv: $(DATA_DIR)/selected.csv
	@uv run python -m src.feature_scaling $(DATA_DIR)

scrape:
	@uv run python -m src.scrape "weekly"

feature_extraction: $(DATA_DIR)/extracted.csv
feature_selection: $(DATA_DIR)/selected.csv
feature_scaling: $(DATA_DIR)/scaled.csv

train: $(DATA_DIR)/scaled.csv
	@mkdir -p $(MODEL_DIR)/$(TODAY)
	@uv run python -m src.train $(TODAY)


test: 
	@mkdir -p $(TEST_DIR)
	@uv run python -m src.feature_extraction $(TEST_DIR) test
	@uv run python -m src.feature_selection $(TEST_DIR)
	@uv run python -m src.feature_scaling $(TEST_DIR)

	@uv run python -m src.test $(TEST_DIR)