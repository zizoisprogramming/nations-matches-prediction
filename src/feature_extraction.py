
import json
import requests
import time
import math
import asyncio
import nest_asyncio
import argparse

import pandas as pd
import numpy as np
import datetime as dt

from pathlib import Path
from countryinfo import CountryInfo
from geopy.exc import GeocoderRateLimited
from geopy.geocoders import Nominatim
from playwright.async_api import async_playwright

from src.helpers.cache import load_cache, save_cache
from src.helpers.apis import _api_get
from src.helpers.helpers import _safe_ratio, slim_event
from src.helpers.constants import NEW_DATA_PATH, TEST_DATA_PATH

nest_asyncio.apply()


BASE_DIR = Path(__file__).parent.parent

CACHE_DIR = BASE_DIR / "data" / "cache"
COORDS_CACHE_PATH = CACHE_DIR / "coords_cache.json"
EVENTS_CACHE_PATH = CACHE_DIR / "team_events.json"
WEATHER_CACHE_PATH = CACHE_DIR / "weather_cache.json"
RATINGS_CACHE_PATH = CACHE_DIR / "ratings_cache.json"
CAPITALS_CACHE_PATH = CACHE_DIR / "capitals_cache.json"
TEAM_IDS_PATH = CACHE_DIR / "team_ids.json"

N_MATCHES = 2

class FeatureExtraction():

    def __init__(self):
        self._geolocator = Nominatim(
            user_agent="nations-matches-prediction",
            timeout=10
        )

        self.coords_cache = load_cache(COORDS_CACHE_PATH)
        self.capitals_cache = load_cache(CAPITALS_CACHE_PATH)
        self.team_ids = load_cache(TEAM_IDS_PATH)
        self.ratings_cache = load_cache(RATINGS_CACHE_PATH)
        self.events_cache = load_cache(EVENTS_CACHE_PATH)

    def _haversine(self, lat1, lon1, lat2, lon2):

        lat1, lon1, lat2, lon2 = (
            np.radians(pd.to_numeric(v, errors="coerce")) for v in (lat1, lon1, lat2, lon2)
        )
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    def _get_capital(self, team_name: str, cache: dict) -> str | None:
        if team_name in cache:
            return cache[team_name]
        print(f"{team_name} not found in cache")
        try:
            cache[team_name] = CountryInfo(team_name).capital()
            save_cache(CAPITALS_CACHE_PATH, cache)
            return cache[team_name]
        except Exception as e:
            raise e

    def _geocode_place(self, place: str, cache: dict) -> tuple | None:
        if place in cache:
            return tuple(cache[place]) 
        print(f"{place} not found in cache")
        for _ in range(5):
            try:
                time.sleep(1.2)  # Nominatim ~1 req/sec
                location = self._geolocator.geocode(place)
                result = (location.latitude, location.longitude) if location else None
                cache[place] = result
                save_cache(COORDS_CACHE_PATH, cache)
                return result
            except GeocoderRateLimited:
                time.sleep(60)
            except Exception:
                time.sleep(5)
        raise Exception(f"Couldn't get geocode for {place}")

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

        return df

    def _fetch_weather(self, lat, lon, date: str, cache: dict) -> dict | None:
        key = f"{lat}_{lon}_{date}"
        if key in cache:
            return cache[key]
        print(f"{key} not found in cache")
        for _ in range(5):
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
                save_cache(WEATHER_CACHE_PATH, cache)
                return result
            except Exception:
                time.sleep(5)
        raise Exception(f"Couldn't find weather fot {lat, lon, date}")

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch historical weather for home/away capitals + stadium; add raw temp/wind columns."""
        df = df.copy()
        cache = load_cache(WEATHER_CACHE_PATH)
        match_date = pd.to_datetime(df["date"]).dt.date.astype(str)

        for prefix, lat_col, lon_col in [
            ("home", "home_lat", "home_lon"),
            ("away", "away_lat", "away_lon"),
            ("stadium", "stadium_lat", "stadium_lon"),
        ]:
            temp_max, temp_min, wind, precipitation = [], [], [], []
            for lat, lon, date in zip(df[lat_col], df[lon_col], match_date):
                w = self._fetch_weather(lat, lon, date, cache) if pd.notna(lat) and pd.notna(lon) else None
                if not w:
                    raise Exception(f"Couldn't find weather for {lat, lon, date}")
                
                temp_max.append(w["temperature_max"] if w else np.nan)
                temp_min.append(w["temperature_min"] if w else np.nan)
                wind.append(w["wind_speed"] if w else np.nan)
                precipitation.append(w["precipitation"] if w else np.nan)

            df[f"{prefix}_temperature_max"] = temp_max
            df[f"{prefix}_temperature_min"] = temp_min
            df[f"{prefix}_wind_speed"] = wind
            df[f"{prefix}_precipitation"] = precipitation

        return df

    async def _make_browser_session(self):
        
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

    async def _search_team_id(self, page, team_name: str) -> int | None:

        url = f"https://www.sofascore.com/api/v1/search/all?q={team_name.replace(' ', '%20')}&page=0"
        data = await _api_get(page, url)
        if not data:
            raise Exception(f"Couldn't get team id for {team_name}")
        for r in data.get("results", []):
            if r.get("type") != "team":
                continue
            entity = r["entity"]
            if entity.get("sport", {}).get("slug") != "football":
                continue
            if entity.get("national") and team_name.lower() in entity.get("name", "").lower():
                return entity["id"]
        # for r in data.get("results", []):
        #     if r.get("type") == "team" and r["entity"].get("sport", {}).get("slug") == "football":
        #         return r["entity"]["id"]
        raise Exception(f"Couldn't get team id for {team_name}")

    async def _get_team_id(self, page, team_name: str, team_ids: dict) -> int | None:
        if team_name in team_ids:
            return team_ids[team_name]
        print(f"{team_name} not found in cache")
        team_id = await self._search_team_id(page, team_name)
        if team_id:
            team_ids[team_name] = team_id
            save_cache(TEAM_IDS_PATH, team_ids)
        return team_id


    async def _fetch_recent_finished_events(self, team_id: int, before_ts: int):
        if f"{team_id}" not in self.events_cache:
            raise Exception(f"Couldn't find events for {team_id}")
        events = self.events_cache[f"{team_id}"]
        new_events = []

        for e in events:
            if e["startTimestamp"] < before_ts:
                new_events.append(e)

        new_events.sort(key=lambda e: e.get("startTimestamp", 0), reverse=True)
        return new_events[:N_MATCHES]

    async def _fetch_finished_events_till_overlap(self, page, team_id: int, max_pages=10):

        if f"{team_id}" in self.events_cache:
            events = self.events_cache[f"{team_id}"]
        else:
            events = []

        cached_ids = {e.get("id") for e in events}
        overlap = False
        for page_num in range(max_pages):
            data = await _api_get(page, f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/{page_num}")
            await asyncio.sleep(1.2)
            if not data:
                break

            for e in data.get("events", []):
                new_event = slim_event(e)
                
                if new_event.get("id") in cached_ids:
                    overlap = True
                    continue

                if new_event["status"] == "finished" and new_event["hasStats"]:
                    events.append(new_event)
                
            if not data.get("hasNextPage", False) or overlap:
                break

        events.sort(key=lambda e: e.get("startTimestamp", 0), reverse=True)
        self.events_cache[f"{team_id}"] = events
        save_cache(EVENTS_CACHE_PATH, self.events_cache)
    
    async def _lineup_stats(self, page, event: dict, team_id: int):
        if not event.get("hasStats", False):
            return {"ranking": None, "rating": None, "shots": None, "scored": None, "scored_against": None, "shots_against": None}
        data = await _api_get(page, f"https://www.sofascore.com/api/v1/event/{event['id']}/lineups")
        await asyncio.sleep(1.2)
        if not data:
            return {"ranking": None, "rating": None, "shots": None, "scored": None, "scored_against": None, "shots_against": None}

        our_side = "home" if event.get("homeTeamId") == team_id else "away"
        opp_side = "away" if our_side == "home" else "home"

        scored = event.get("homeScore") if our_side == "home" else event.get("awayScore")
        scored_against = event.get("homeScore") if our_side == "away" else event.get("awayScore")
        ranking = event.get(f"{our_side}Ranking")

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
                shots += stats.get("totalShots") or 0
            avg = round(sum(ratings) / len(ratings), 3) if ratings else None
            return avg, (shots)

        our_rating, our_shots = extract(our_side)
        _, opp_shots = extract(opp_side)
        return {"ranking": ranking, "rating": our_rating, "shots": our_shots, "scored": scored, "scored_against": scored_against, "shots_against": opp_shots}

    async def _last_n_form_stats(self, page, team_id: int, before_ts: int, cache: dict, mode="no-pull") -> dict:

        if mode == "pull" or f"{team_id}" not in self.events_cache:
            await self._fetch_finished_events_till_overlap(page, team_id)

        events = await self._fetch_recent_finished_events(team_id, before_ts)

        ranking = None
        fixes, shots, scored, shots_against, rel_shots, rel_goals = [], [], [], [], [], []

        cache_key = f"{team_id}_{before_ts}"
        if cache_key in cache:
            stats = cache[cache_key]
            fixes = stats["fix"]
            shots = stats["shots"]
            scored = stats["scored"]
            shots_against = stats["shots_against"]
            rel_shots = stats["relative_shots"]
            rel_goals = stats["relative_goals"]
            ranking = stats["ranking"]
        else:
            print(f"{cache_key} not found in cache")
            done = 0
            for event in events:
                
                lineup = await self._lineup_stats(page, event, team_id)
                if any([v is None for _, v in lineup.items()]): continue
                stats = {
                    "fix": lineup["rating"],
                    "shots_against": lineup["shots_against"],
                    "shots": lineup["shots"],
                    "scored": lineup["scored"],
                    "ranking": lineup["ranking"],
                    "relative_shots": _safe_ratio(lineup["shots"], lineup["shots_against"]),
                    "relative_goals": _safe_ratio(lineup["scored"], lineup["scored_against"]),
                }

                fixes.append(stats["fix"])
                shots.append(stats["shots"])
                scored.append(stats["scored"])
                shots_against.append(stats["shots_against"])
                rel_shots.append(stats["relative_shots"])
                rel_goals.append(stats["relative_goals"])
                ranking = stats["ranking"]

                done += 1
                if done >= N_MATCHES: break
            if done < N_MATCHES:
                raise Exception(f"Couldn't find {N_MATCHES} matches for {team_id}")
            
            cache[cache_key] = {
                "fix": fixes,
                "shots_against": shots_against,
                "shots": shots,
                "scored": scored,
                "relative_shots": rel_shots,
                "relative_goals": rel_goals,
                "ranking": ranking
            }
            save_cache(RATINGS_CACHE_PATH, cache)


        to_return_obj = {}
        to_return_obj["ranking"] = ranking
        for i in range(len(fixes)):
            to_return_obj[f"fix_{i + 1}"] = fixes[i]
            to_return_obj[f"shots_{i + 1}"] = shots[i]
            to_return_obj[f"scored_{i + 1}"] = scored[i]
            to_return_obj[f"shots_against_{i + 1}"] = shots_against[i]
            to_return_obj[f"relative_shots_{i + 1}"] = rel_shots[i]
            to_return_obj[f"relative_goals_{i + 1}"] = rel_goals[i]

        return to_return_obj

    async def _fetch_sofascore_features_async(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        playwright, browser, page = await self._make_browser_session()
        to_drop = []
        try:
            
            for idx, row in df.iterrows():
                match_date = pd.to_datetime(row["date"]).date()
                before_ts = int(dt.datetime(match_date.year, match_date.month, match_date.day).timestamp())
                
                for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
                    team_id = await self._get_team_id(page, str(row[team_col]).strip(), self.team_ids)
                    if team_id is None:
                        continue
                    try:
                        stats = await self._last_n_form_stats(page, team_id, before_ts, self.ratings_cache)
                        for k, v in stats.items():
                            col_name = f"{prefix}_ranking" if k == "ranking" else f"{prefix}_{k}"
                            df.at[idx, col_name] = v
                    except Exception as e:
                        print(e)
                        to_drop.append(idx)

            df = df.drop(index=to_drop).reset_index(drop=True)
            with open("to_drop.json", "w") as f:
                json.dump(to_drop, f)
        finally:
            await browser.close()
            await playwright.stop()

        return df

    def _add_sofascore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return asyncio.get_event_loop().run_until_complete(self._fetch_sofascore_features_async(df))

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered columns that may be absent from raw input."""
        df = df.copy()

        if 'home_stadium_distance_km' not in df.columns:
            df['home_stadium_distance_km'] = self._haversine(
                df["home_lat"],
                df["home_lon"],
                df["stadium_lat"],
                df["stadium_lon"],
            )
        if 'away_stadium_distance_km' not in df.columns:
            df['away_stadium_distance_km'] = self._haversine(
                df["away_lat"],
                df["away_lon"],
                df["stadium_lat"],
                df["stadium_lon"],
            )
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

    def run(self, path, save_dir):
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            X = df.drop(columns=['result'], errors='ignore').copy()
            X = self._add_location_features(X)
            X = self._add_weather_features(X)
            X = self._add_sofascore_features(X)
            X = self._add_derived_features(X)
            X.to_csv(f"{save_dir}/extracted.csv", index=False)
            return f"{save_dir}/extracted.csv"
        except Exception as e:
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir")
    parser.add_argument("mode")

    args = parser.parse_args()
    save_dir = args.save_dir
    mode = args.mode

    fe = FeatureExtraction()
    if mode == "test":
        fe.run(TEST_DATA_PATH, save_dir)
    elif mode == "train":
        fe.run(NEW_DATA_PATH, save_dir)
    else:
        raise Exception("Unknown mode")
