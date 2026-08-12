# Nations Matches Prediction

A machine learning pipeline for predicting the outcomes of international football matches across major global tournaments, including the FIFA World Cup, continental championships, and Nations League competitions.

## Overview

This project scrapes historical match data, engineers rich feature sets from multiple data sources (geographic, weather, team form, and rankings), and trains an ensemble model to predict match results as **home win**, **draw**, or **away win**.

## Target Competitions

- FIFA World Cup
- UEFA European Championship
- UEFA Nations League
- CONMEBOL Copa America
- AFC Asian Cup
- Concacaf Nations League
- CAF Africa Cup of Nations

## Pipeline

The project follows a modular ML pipeline orchestrated via `Makefile`:

```
Scrape → Feature Extraction → Feature Selection → Feature Scaling → Train / Inference
```

### 1. Scraping (`src/scrape.py`)

Scrapes upcoming and recent matches from **FIFA Match Centre** and **ESPN fixtures** using Playwright with anti-bot measures (throttling, retries, user-agent rotation). Matches are filtered by target competition using fuzzy text matching, then enriched with venue and attendance data.

```bash
make scrape
```

### 2. Feature Extraction (`src/feature_extraction.py`)

Enriches scraped match data with features from multiple sources:

- **Geographic features**: Haversine distance between national capitals and the stadium venue (via `geopy` and `countryinfo`)
- **Weather features**: Historical temperature, precipitation, and wind speed for capitals and stadium location (via Open-Meteo archive API)
- **Team form & rankings**: Last N match ratings, shots, goals conceded, and world rankings (via SofaScore API)
- **Derived features**: Ranking differences, temperature averages, relative shot/goal ratios, cyclic day encoding

```bash
make feature_extraction
```

### 3. Feature Selection (`src/feature_selection.py`)

Selects a curated set of ~40 predictive features from the full engineered feature space, including distances, team form metrics, rankings, weather conditions, and temporal features.

```bash
make feature_selection
```

### 4. Feature Scaling (`src/feature_scaling.py`)

Applies a pre-trained scaler to numeric features and encodes cyclic temporal features (`day_sin`, `day_cos`).

```bash
make feature_scaling
```

### 5. Model Training (`src/train.py`)

Trains a **VotingClassifier** ensemble with grid search cross-validation:

- **XGBoost** (`XGBClassifier`)
- **Logistic Regression**
- **Support Vector Machine** (`SVC`)

Class weights are balanced to handle outcome imbalance. The best model is saved to `models/`.

```bash
make train
```

### 6. Inference (`src/inference.py`)

Loads the trained model and produces:

- Match outcome predictions (`home_win` / `draw` / `away_win`)
- Class probabilities (`prob_home_win`, `prob_draw`, `prob_away_win`)

```bash
make test
```

## Full Training Pipeline

```bash
make train
```

## Project Structure

```
nations-matches-prediction/
├── Makefile                 # Pipeline orchestration
├── pyproject.toml           # Project dependencies
├── main.py                  # Entry point
├── src/
│   ├── scrape.py            # Match data collection
│   ├── feature_extraction.py # Feature engineering
│   ├── feature_selection.py # Feature subset selection
│   ├── feature_scaling.py   # Scaling & encoding
│   ├── train.py             # Model training
│   ├── test.py              # Model evaluation
│   ├── inference.py         # Prediction utilities
│   └── helpers/             # Caching, API utilities, constants
├── models/                  # Saved models
├── data/
│   ├── cache/               # Cached API responses & geocoding
│   ├── staging/             # Raw scraped data
│   ├── train/               # Training splits
│   └── test/                # Test splits
└── notebooks/               # Exploratory analysis notebooks
```

## Dependencies

- Python >= 3.14
- Playwright / Selenium (browser automation)
- XGBoost, Scikit-learn (modeling)
- pandas, numpy (data processing)
- geopy, countryinfo (geographic data)
- requests, Open-Meteo (weather API)
- joblib (model persistence)

Install dependencies with:

```bash
uv sync
```

## License

MIT
