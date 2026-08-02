
import json
import requests
import time
import math
import asyncio
import nest_asyncio

import pandas as pd
import numpy as np
import datetime as dt

from pathlib import Path
from countryinfo import CountryInfo
from geopy.exc import GeocoderRateLimited
from geopy.geocoders import Nominatim
from playwright.async_api import async_playwright

from src.helpers.cache import _load_cache, _save_cache
from src.helpers.apis import _api_get
from src.helpers.helpers import _safe_ratio

nest_asyncio.apply()


BASE_DIR = Path(__file__).parent.parent

CACHE_DIR = BASE_DIR / "data" / "cache"
COORDS_CACHE_PATH = CACHE_DIR / "coords_cache.json"
WEATHER_CACHE_PATH = CACHE_DIR / "weather_cache.json"
RATINGS_CACHE_PATH = CACHE_DIR / "ratings_cache.json"
CAPITALS_CACHE_PATH = CACHE_DIR / "capitals_cache.json"
TEAM_IDS_PATH = CACHE_DIR / "team_ids.json"

class FeatureExtraction():

    def __init__(self):
        self._geolocator = Nominatim(
            user_agent="nations-matches-prediction",
            timeout=10
        )

        self.coords_cache = _load_cache(COORDS_CACHE_PATH)
        self.capitals_cache = _load_cache(CAPITALS_CACHE_PATH)
        self.team_ids = _load_cache(TEAM_IDS_PATH)
        self.ratings_cache = _load_cache(RATINGS_CACHE_PATH)

    def _haversine(lat1, lon1, lat2, lon2):

        lat1, lon1, lat2, lon2 = (
            np.radians(pd.to_numeric(v, errors="coerce")) for v in (lat1, lon1, lat2, lon2)
        )
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    def _get_capital(team_name: str, cache: dict) -> str | None:
        if team_name in cache:
            return cache[team_name]
        try:
            return CountryInfo(team_name).capital()
        except Exception as e:
            raise e


    def _geocode_place(self, place: str, cache: dict) -> tuple | None:
        if place in cache:
            return tuple(cache[place]) if cache[place] else None
        for attempt in range(5):
            try:
                time.sleep(1.2)  # Nominatim ~1 req/sec
                location = self._geolocator.geocode(place)
                result = (location.latitude, location.longitude) if location else None
                cache[place] = result
                _save_cache(COORDS_CACHE_PATH, cache)
                return result
            except GeocoderRateLimited:
                time.sleep(60)
            except Exception:
                time.sleep(5)
        cache[place] = None
        _save_cache(COORDS_CACHE_PATH, cache)
        return None

    def _add_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geocode home/away capitals + stadium city; add lat/lon and away_stadium_distance_km."""
        df = df.copy()

        def _capitals(teams):
            out = []
            for t in teams:
                cap = self._get_capital(t, self.capitals_cache)
                out.append(f"{cap}, {t}" if cap else None)
            return out

        home_places = _capitals(df["home_team"])
        away_places = _capitals(df["away_team"])
        stadium_places = (df["city"].astype(str) + ", " + df["country"].astype(str)).tolist()

        def _geocode_list(places):
            coords = [self._geocode_place(p, self.coords_cache) if p else None for p in places]
            lat = [c[0] if c else np.nan for c in coords]
            lon = [c[1] if c else np.nan for c in coords]
            return lat, lon

        df["home_lat"], df["home_lon"] = _geocode_list(home_places)
        df["away_lat"], df["away_lon"] = _geocode_list(away_places)
        df["stadium_lat"], df["stadium_lon"] = _geocode_list(stadium_places)

        if "away_stadium_distance_km" not in df.columns:
            df["away_stadium_distance_km"] = self._haversine(
                df["away_lat"], df["away_lon"], df["stadium_lat"], df["stadium_lon"]
            )
        return df



    def _fetch_weather(lat, lon, date: str, cache: dict) -> dict | None:
        key = f"{lat}_{lon}_{date}"
        if key in cache:
            return cache[key]
        for attempt in range(5):
            try:
                time.sleep(1.5)  # well under Open-Meteo's 600/min limit
                r = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": date,
                        "end_date": date,
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                        "timezone": "auto",
                    },
                    timeout=15,
                )
                r.raise_for_status()
                daily = r.json()["daily"]
                result = {
                    "temperature_max": daily["temperature_2m_max"][0],
                    "temperature_min": daily["temperature_2m_min"][0],
                    "precipitation": daily["precipitation_sum"][0],
                    "wind_speed": daily["wind_speed_10m_max"][0],
                }
                cache[key] = result
                _save_cache(WEATHER_CACHE_PATH, cache)
                return result
            except Exception:
                time.sleep(5)
        cache[key] = None
        _save_cache(WEATHER_CACHE_PATH, cache)
        return None


    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch historical weather for home/away capitals + stadium; add raw temp/wind columns."""
        df = df.copy()
        cache = _load_cache(WEATHER_CACHE_PATH)
        match_date = pd.to_datetime(df["date"]).dt.date.astype(str)

        for prefix, lat_col, lon_col in [
            ("home", "home_lat", "home_lon"),
            ("away", "away_lat", "away_lon"),
            ("stadium", "stadium_lat", "stadium_lon"),
        ]:
            temp_max, temp_min, wind = [], [], []
            for lat, lon, date in zip(df[lat_col], df[lon_col], match_date):
                w = self._fetch_weather(lat, lon, date, cache) if pd.notna(lat) and pd.notna(lon) else None
                temp_max.append(w["temperature_max"] if w else np.nan)
                temp_min.append(w["temperature_min"] if w else np.nan)
                wind.append(w["wind_speed"] if w else np.nan)
            df[f"{prefix}_temperature_max"] = temp_max
            df[f"{prefix}_temperature_min"] = temp_min
            df[f"{prefix}_wind_speed"] = wind

        return df


    async def _make_browser_session():
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.sofascore.com/",
            },
        )
        page = await context.new_page()
        await page.goto("https://www.sofascore.com/", wait_until="domcontentloaded")
        await asyncio.sleep(2)
        return playwright, browser, page



    async def _search_team_id(page, team_name: str) -> int | None:

        url = f"https://www.sofascore.com/api/v1/search/all?q={team_name.replace(' ', '%20')}&page=0"
        data = await _api_get(page, url)
        if not data:
            return None
        for r in data.get("results", []):
            if r.get("type") != "team":
                continue
            entity = r["entity"]
            if entity.get("sport", {}).get("slug") != "football":
                continue
            if entity.get("national") and team_name.lower() in entity.get("name", "").lower():
                return entity["id"]
        for r in data.get("results", []):
            if r.get("type") == "team" and r["entity"].get("sport", {}).get("slug") == "football":
                return r["entity"]["id"]
        return None


    async def _get_team_id(self, page, team_name: str, team_ids: dict) -> int | None:
        if team_name in team_ids:
            return team_ids[team_name]
        team_id = await self._search_team_id(page, team_name)
        if team_id:
            team_ids[team_name] = team_id
            _save_cache(TEAM_IDS_PATH, team_ids)
        return team_id


    async def _fetch_recent_finished_events(page, team_id: int, before_ts: int, max_events=3, max_pages=3):
        events = []
        for page_num in range(max_pages):
            data = await _api_get(page, f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/{page_num}")
            await asyncio.sleep(0.8)
            if not data:
                break
            for e in data.get("events", []):
                if e.get("status", {}).get("type") != "finished":
                    continue
                if e.get("startTimestamp", 0) >= before_ts:
                    continue
                events.append(e)
            if len(events) >= max_events or not data.get("hasNextPage", False):
                break
        events.sort(key=lambda e: e.get("startTimestamp", 0), reverse=True)
        return events[:max_events]



    async def _lineup_stats(page, event: dict, team_id: int):
        home = event.get("homeTeam", {})
        if not event.get("hasEventPlayerStatistics", False):
            return {"rating": None, "shots": None, "shots_against": None}
        data = await _api_get(page, f"https://www.sofascore.com/api/v1/event/{event['id']}/lineups")
        await asyncio.sleep(0.5)
        if not data:
            return {"rating": None, "shots": None, "shots_against": None}

        our_side = "home" if home.get("id") == team_id else "away"
        opp_side = "away" if our_side == "home" else "home"

        def extract(side):
            players = data.get(side, {}).get("players", [])
            ratings, shots = [], 0
            for p in players:
                if p.get("substitute", True):
                    continue
                stats = p.get("statistics", {})
                r = stats.get("rating")
                if r is not None:
                    try:
                        r = float(r)
                        if not math.isnan(r):
                            ratings.append(r)
                    except Exception:
                        pass
                shots += stats.get("totalShots", 0) or 0
            avg = round(sum(ratings) / len(ratings), 3) if ratings else None
            return avg, (shots if shots > 0 else None)

        our_rating, our_shots = extract(our_side)
        _, opp_shots = extract(opp_side)
        return {"rating": our_rating, "shots": our_shots, "shots_against": opp_shots}


    async def _last_2_form_stats(self, page, team_id: int, before_ts: int, cache: dict) -> dict:
        """Pulls last 3 finished matches, keeps the first 2 that have lineup data
        (mirrors the _1/_2/_3 -> keep-best-2 cleanup done in feature-engineering.ipynb)."""
        events = await self._fetch_recent_finished_events(page, team_id, before_ts, max_events=3)
        ranking = None
        fixes, shots_against, rel_shots = [], [], []

        for event in events:
            cache_key = f"{event['id']}_{team_id}"
            if cache_key in cache:
                stats = cache[cache_key]
            else:
                lineup = await self._lineup_stats(page, event, team_id)
                stats = {
                    "rating": lineup["rating"],
                    "shots_against": lineup["shots_against"],
                    "relative_shots": _safe_ratio(lineup["shots"], lineup["shots_against"]),
                }
                cache[cache_key] = stats
                _save_cache(RATINGS_CACHE_PATH, cache)

            if ranking is None:
                home = event.get("homeTeam", {})
                ranking = home.get("ranking") if home.get("id") == team_id else event.get("awayTeam", {}).get("ranking")

            fixes.append(stats["rating"])
            shots_against.append(stats["shots_against"])
            rel_shots.append(stats["relative_shots"])

        keep_idx = [i for i, v in enumerate(fixes) if v is not None][:2]
        if len(keep_idx) < 2:
            keep_idx = list(range(min(2, len(fixes))))
        while len(keep_idx) < 2:
            keep_idx.append(None)

        def pick(values):
            return [values[i] if i is not None and i < len(values) else None for i in keep_idx]

        fx, sa, rs = pick(fixes), pick(shots_against), pick(rel_shots)
        return {
            "fix_1": fx[0], "fix_2": fx[1],
            "shots_against_1": sa[0], "shots_against_2": sa[1],
            "relative_shots_1": rs[0], "relative_shots_2": rs[1],
            "ranking": ranking,
        }


    async def _fetch_sofascore_features_async(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        

        playwright, browser, page = await self._make_browser_session()
        try:
            for idx, row in df.iterrows():
                match_date = pd.to_datetime(row["date"]).date()
                before_ts = int(dt.datetime(match_date.year, match_date.month, match_date.day).timestamp())

                for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
                    team_id = await self._get_team_id(page, str(row[team_col]).strip(), self.team_ids)
                    if team_id is None:
                        continue
                    stats = await self._last_2_form_stats(page, team_id, before_ts, self.ratings_cache)
                    for k, v in stats.items():
                        col_name = f"{prefix}_ranking" if k == "ranking" else f"{prefix}_{k}"
                        df.at[idx, col_name] = v
        finally:
            await browser.close()
            await playwright.stop()

        return df


    def _add_sofascore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return asyncio.get_event_loop().run_until_complete(self._fetch_sofascore_features_async(df))


    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered columns that may be absent from raw input."""
        df = df.copy()

        if 'home_stadium_wind_speed' not in df.columns:
            df['home_stadium_wind_speed'] = np.abs(
                df['home_wind_speed'] - df['stadium_wind_speed']
            )
        if 'away_stadium_wind_speed' not in df.columns:
            df['away_stadium_wind_speed'] = np.abs(
                df['away_wind_speed'] - df['stadium_wind_speed']
            )
        if 'ranking_diff' not in df.columns:
            df['ranking_diff'] = df['home_ranking'] - df['away_ranking']
        if 'stadium_temperature_avg' not in df.columns:
            df['stadium_temperature_avg'] = (
                df['stadium_temperature_max'] + df['stadium_temperature_min']
            ) / 2
        if 'home_temperature_avg' not in df.columns:
            df['home_temperature_avg'] = (
                df['home_temperature_max'] + df['home_temperature_min']
            ) / 2
        if 'away_temperature_avg' not in df.columns:
            df['away_temperature_avg'] = (
                df['away_temperature_max'] + df['away_temperature_min']
            ) / 2
        if 'home_stadium_temp_avg' not in df.columns:
            home_max = np.abs(df['home_temperature_max'] - df['stadium_temperature_max'])
            home_min = np.abs(df['home_temperature_min'] - df['stadium_temperature_min'])
            df['home_stadium_temp_avg'] = (home_max + home_min) / 2
        if 'away_stadium_temp_avg' not in df.columns:
            away_max = np.abs(df['away_temperature_max'] - df['stadium_temperature_max'])
            away_min = np.abs(df['away_temperature_min'] - df['stadium_temperature_min'])
            df['away_stadium_temp_avg'] = (away_max + away_min) / 2

        return df

    def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Cyclic encoding for day and month columns."""
        df = df.copy()
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df.drop(columns=['day', 'month'], errors='ignore')

    def run(self, path, save_dir):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        X = df.drop(columns=['result'], errors='ignore').copy()
        X = self._add_location_features(X)
        X = self._add_weather_features(X)
        X = self._add_sofascore_features(X)
        X = self._add_cyclic_features(X)
        X.to_csv(f"{save_dir}/extracted.csv")
