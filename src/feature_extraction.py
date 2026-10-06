import sys

import json
import unicodedata
import re
import secrets
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
from difflib import SequenceMatcher
from countryinfo import CountryInfo
from geopy.exc import GeocoderRateLimited
from geopy.geocoders import Nominatim
from playwright.async_api import async_playwright

from src.helpers.cache import load_cache, save_cache
from src.helpers.apis import _api_get, SofascoreBlocked, RequestFailed
from src.helpers.helpers import _safe_ratio, slim_event
from src.helpers.constants import NEW_DATA_PATH, TEST_DATA_PATH, FORM_SOURCE, APIFOOTBALL_MATCHES_PATH, FORM_SEED_PATH, MAX_MISSING_FORM_GAMES

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

# Youth, Olympic and other non-senior sides that share a country's name on Sofascore.
_NON_SENIOR_TEAM = re.compile(r"U\d\d|Olympic|Beach|Futsal|Women", re.IGNORECASE)


def _normalize_team_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c)).lower()
    name = name.replace("&", " and ")
    name = re.sub(r"\bst\b\.?", "saint", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return " ".join(name.split())


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
        # Teams whose events were refreshed from Sofascore this run, and
        # (team_id, event_id) pairs whose lineups gave full stats.
        self._refreshed = set()
        self._verified = set()

    def _haversine(self, lat1, lon1, lat2, lon2):

        lat1, lon1, lat2, lon2 = (
            np.radians(pd.to_numeric(v, errors="coerce")) for v in (lat1, lon1, lat2, lon2)
        )
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    def _get_capital(self, team_name: str, cache: dict) -> str | None:
        """
        Capital for a team's country. If countryinfo doesn't know the name, try "St"
        spelled out, then other names in team_ids.json with the same Sofascore id
        (e.g. "Congo DR" -> "DR Congo"), preferring ones with the same words.
        Returns None (the match is dropped later) instead of failing the run.
        """
        if cache.get(team_name):
            return cache[team_name]
        print(f"{team_name} not found in cache")

        candidates = [team_name, re.sub(r"\bSt\.?\s", "Saint ", team_name)]
        team_id = self.team_ids.get(team_name)
        if team_id is not None:
            same_id = [n for n, i in self.team_ids.items() if i == team_id and n != team_name]
            words = set(_normalize_team_name(team_name).split())
            same_id.sort(key=lambda n: set(_normalize_team_name(n).split()) != words)
            candidates += same_id

        for name in dict.fromkeys(candidates):
            capital = cache.get(name)
            if not capital:
                try:
                    capital = CountryInfo(name).capital()
                except Exception:
                    continue
            if capital:
                if name != team_name:
                    print(f"Capital for '{team_name}' taken from '{name}': {capital}")
                cache[team_name] = capital
                save_cache(CAPITALS_CACHE_PATH, cache)
                return capital
        print(f"Couldn't find a capital for '{team_name}'")
        return None

    def _geocode_place(self, place: str, cache: dict) -> tuple | None:
        """
        Coordinates for "city, country". If Nominatim doesn't know the place, retry with
        "St" spelled out ("Basseterre, Saint Kitts and Nevis"), then fall back to just
        the country. Only real results are cached, so failures are retried next run.
        """
        if cache.get(place):
            return tuple(cache[place])
        print(f"{place} not found in cache")

        candidates = [place, re.sub(r"\bSt\.?\s", "Saint ", place)]
        if "," in place:
            country = place.rsplit(",", 1)[1].strip()
            candidates += [country, re.sub(r"\bSt\.?\s", "Saint ", country)]

        for query in dict.fromkeys(candidates):  # dedupe, keep order
            result = self._geocode_query(query)
            if result:
                if query != place:
                    print(f"Geocoded '{place}' using '{query}'")
                cache[place] = result
                save_cache(COORDS_CACHE_PATH, cache)
                return result
        print(f"Couldn't geocode '{place}'")
        return None

    def _geocode_query(self, query: str) -> tuple | None:
        for _ in range(5):
            try:
                time.sleep(1.2)  # Nominatim ~1 req/sec
                location = self._geolocator.geocode(query)
                return (location.latitude, location.longitude) if location else None
            except GeocoderRateLimited:
                time.sleep(60)
            except Exception:
                time.sleep(5)
        raise Exception(f"Couldn't get geocode for {query}")

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

        # Without coordinates there's no distance or weather: drop the match instead of
        # failing the whole run.
        coord_cols = ["home_lat", "home_lon", "away_lat", "away_lon", "stadium_lat", "stadium_lon"]
        missing = df[coord_cols].isna().any(axis=1)
        for _, r in df[missing].iterrows():
            print(f"Dropping {r['home_team']} vs {r['away_team']} ({r['date']}): no coordinates")
        return df[~missing].reset_index(drop=True)

    def _fetch_weather(self, lat, lon, date: str, cache: dict) -> dict | None:
        key = f"{lat}_{lon}_{date}"
        if key in cache:
            return cache[key]
        print(f"{key} not found in cache")

        # The archive has no data for the last few days or the future (e.g. predicting
        # tomorrow's match), so recent dates use the forecast API. Those values can still
        # change, so they aren't cached.
        recent = dt.date.fromisoformat(date) >= dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=5)
        url = ("https://api.open-meteo.com/v1/forecast" if recent
               else "https://archive-api.open-meteo.com/v1/archive")
        for _ in range(5):
            try:
                time.sleep(1.5)  # well under Open-Meteo's 600/min limit
                r = requests.get(
                    url,
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
                if any(v is None for v in result.values()):
                    raise ValueError(f"incomplete weather from {url}: {result}")
                if not recent:
                    cache[key] = result
                    save_cache(WEATHER_CACHE_PATH, cache)
                return result
            except Exception:
                time.sleep(5)
        raise Exception(f"Couldn't find weather for {lat, lon, date}")

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
        # No user_agent override: a spoofed UA that doesn't match the real browser's
        # fingerprint gets every Sofascore API call rejected with 403.
        context = await browser.new_context(
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.sofascore.com/",
                # Sofascore answers API requests without it with a Cloudflare 403
                # "challenge"; the value isn't checked.
                "X-Requested-With": secrets.token_hex(3),
            },
        )
        page = await context.new_page()
        # Visit the homepage first so the session looks like a normal browser. Only the
        # first response is needed: the full page can take 30s+ on a slow connection,
        # and the API works without it, so a failure here isn't fatal.
        for attempt in range(1, 4):
            try:
                await page.goto("https://www.sofascore.com/", wait_until="commit", timeout=60_000)
                break
            except Exception as e:
                print(f"  ⚠️  Sofascore homepage attempt {attempt}/3 failed: {str(e).splitlines()[0]}")
                await asyncio.sleep(5 * attempt)
        await asyncio.sleep(2)
        return playwright, browser, page

    async def _search_team_id(self, page, team_name: str) -> int | None:
        """
        Find the senior men's national team for team_name via Sofascore search.
        Names are compared after normalizing (accents, "&"/"and", "St"/"Saint"), so
        spelling variants like "St Vincent ..." or "Congo DR" still resolve.
        """
        target = _normalize_team_name(team_name)
        queries = [team_name]
        if target != team_name.lower():
            queries.append(target)  # e.g. "St Vincent ..." -> "saint vincent ..."

        seen = []
        for query in queries:
            url = f"https://www.sofascore.com/api/v1/search/all?q={query.replace(' ', '%20')}&page=0"
            data = await _api_get(page, url)
            if not data:
                continue

            best, best_score = None, 0.0
            for r in data.get("results", []):
                entity = r.get("entity", {})
                if (r.get("type") != "team"
                        or entity.get("sport", {}).get("slug") != "football"
                        or not entity.get("national")
                        or entity.get("gender") != "M"
                        or _NON_SENIOR_TEAM.search(entity.get("name", ""))):
                    continue
                name = _normalize_team_name(entity.get("name", ""))
                seen.append(entity.get("name", ""))
                if name == target:
                    score = 1.0
                elif set(name.split()) == set(target.split()):
                    score = 0.95  # same words, different order: "Congo DR" / "DR Congo"
                else:
                    score = SequenceMatcher(None, name, target).ratio()
                if score > best_score:
                    best, best_score = entity, score

            if best is not None and best_score >= 0.85:
                print(f"Resolved team '{team_name}' -> {best['id']} '{best['name']}' (score {best_score:.2f})")
                return best["id"]

        raise Exception(f"Couldn't get team id for {team_name} (candidates: {sorted(set(seen)) or 'none'})")

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
            if e["startTimestamp"] < before_ts and e.get("hasStats", False):
                new_events.append(e)

        new_events.sort(key=lambda e: e.get("startTimestamp", 0), reverse=True)
        # Return a few spare candidates: _last_n_form_stats stops after N_MATCHES usable
        # ones, and a lineup can still come back without ratings/shots.
        return new_events[:N_MATCHES * 5]

    async def _fetch_finished_events_till_overlap(self, page, team_id: int, max_pages=10):

        if f"{team_id}" in self.events_cache:
            events = self.events_cache[f"{team_id}"]
        else:
            events = []

        cached_ids = {e.get("id") for e in events}
        overlap = False
        for page_num in range(max_pages):
            data = await _api_get(page, f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/{page_num}")
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
        # Raises RequestFailed if Sofascore couldn't be reached; None means no lineups exist.
        data = await _api_get(page, f"https://www.sofascore.com/api/v1/event/{event['id']}/lineups", raise_on_failure=True)
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

    async def _last_n_form_stats(self, page, team_id: int, before_ts: int, cache: dict) -> dict:

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
            if team_id not in self._refreshed:
                await self._fetch_finished_events_till_overlap(page, team_id)
                self._refreshed.add(team_id)
            events = await self._fetch_recent_finished_events(team_id, before_ts)

            done = 0
            for event in events:
                
                try:
                    lineup = await self._lineup_stats(page, event, team_id)
                except RequestFailed as e:
                    # Couldn't reach Sofascore for this game; keep it cached and try the next.
                    print(f"  skipping game {event.get('id')} for now: {e}")
                    continue
                if any([v is None for _, v in lineup.items()]):
                    # Useless for form stats: drop it so it can't displace a good cached game.
                    self.events_cache[f"{team_id}"] = [
                        e for e in self.events_cache[f"{team_id}"] if e.get("id") != event.get("id")
                    ]
                    continue
                self._verified.add((team_id, event.get("id")))
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
            
            done_idx = set()
            try:
                for idx, row in df.iterrows():
                    done_idx.add(idx)
                    match_date = pd.to_datetime(row["date"]).date()
                    # UTC midnight, so ratings-cache keys are the same on any machine's time zone.
                    before_ts = int(dt.datetime(match_date.year, match_date.month, match_date.day, tzinfo=dt.timezone.utc).timestamp())
                
                    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
                        try:
                            team_id = await self._get_team_id(page, str(row[team_col]).strip(), self.team_ids)
                            stats = await self._last_n_form_stats(page, team_id, before_ts, self.ratings_cache)
                            for k, v in stats.items():
                                col_name = f"{prefix}_ranking" if k == "ranking" else f"{prefix}_{k}"
                                df.at[idx, col_name] = v
                        except SofascoreBlocked:
                            raise
                        except Exception as e:
                            # Unknown team or missing form stats: drop the match rather than
                            # aborting the whole run or training on empty features.
                            print(e)
                            to_drop.append(idx)
                            break
            except SofascoreBlocked as e:
                # Everything else would fail too: drop the unfinished rows and stop.
                print(f"Sofascore is blocking this machine, stopping: {e}")
                to_drop.extend(i for i in df.index if i not in done_idx or i == idx)

            df = df.drop(index=sorted(set(to_drop))).reset_index(drop=True)
            with open("to_drop.json", "w") as f:
                json.dump(to_drop, f)
        finally:
            await browser.close()
            await playwright.stop()

        return df

    def _add_sofascore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return asyncio.get_event_loop().run_until_complete(self._fetch_sofascore_features_async(df))

    def _trim_events_cache(self, team_ids):
        """
        Keep each team's events from newest to oldest until N_MATCHES of them are known
        to give full stats; older ones are dropped. Newer unverified events are kept too,
        so a game that later turns out null never pushes out the good cached ones.
        Teams without N_MATCHES verified events are left untouched.
        """
        for team_id in team_ids:
            events = sorted(self.events_cache.get(f"{team_id}", []),
                            key=lambda e: e.get("startTimestamp", 0), reverse=True)
            kept, good = [], 0
            for e in events:
                kept.append(e)
                if (team_id, e.get("id")) in self._verified:
                    good += 1
                    if good >= N_MATCHES:
                        break
            if good >= N_MATCHES:
                self.events_cache[f"{team_id}"] = kept
        save_cache(EVENTS_CACHE_PATH, self.events_cache)

    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Last-N-games form features for both teams, from FORM_SOURCE. Either way, a team
        and date whose Sofascore stats are already in the ratings cache keeps them, so
        historical rows are unchanged.
        """
        if FORM_SOURCE == "sofascore":
            return self._add_sofascore_features(df)

        df = df.copy()
        to_drop = []
        for idx, row in df.iterrows():
            match_date = pd.to_datetime(row["date"]).date()
            before_ts = int(dt.datetime(match_date.year, match_date.month, match_date.day, tzinfo=dt.timezone.utc).timestamp())
            for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
                team = str(row[team_col]).strip()
                try:
                    team_id = self.team_ids.get(team)
                    if team_id is None:
                        raise Exception(f"{team} is not in team_ids.json")
                    cached = self.ratings_cache.get(f"{team_id}_{before_ts}")
                    stats = (self._form_stats_from_cache(cached) if cached
                             else self._apifootball_form_stats(team_id, team, before_ts))
                    for k, v in stats.items():
                        col_name = f"{prefix}_ranking" if k == "ranking" else f"{prefix}_{k}"
                        df.at[idx, col_name] = v
                except Exception as e:
                    print(e)
                    to_drop.append(idx)
                    break
        return df.drop(index=sorted(set(to_drop))).reset_index(drop=True)

    @staticmethod
    def _form_stats_from_cache(entry: dict) -> dict:
        """Ratings-cache entry -> the same feature dict _last_n_form_stats returns."""
        out = {"ranking": entry["ranking"]}
        for i in range(len(entry["fix"])):
            out[f"fix_{i + 1}"] = entry["fix"][i]
            out[f"shots_{i + 1}"] = entry["shots"][i]
            out[f"scored_{i + 1}"] = entry["scored"][i]
            out[f"shots_against_{i + 1}"] = entry["shots_against"][i]
            out[f"relative_shots_{i + 1}"] = entry["relative_shots"][i]
            out[f"relative_goals_{i + 1}"] = entry["relative_goals"][i]
        return out

    def _apifootball_games(self) -> dict:
        """Sofascore team id -> its games with form stats, newest first: the daily-collected
        API-Football games plus each team's last 2 games seeded from the Sofascore caches
        (src/seed_form_cache.py), which newer API-Football games push out of the last 2."""
        if getattr(self, "_af_games", None) is None:
            from src.collect_apifootball import team_summary
            games = {}
            for record in load_cache(APIFOOTBALL_MATCHES_PATH).values():
                teams = list(record["teams"].values())
                for side in teams:
                    if side.get("sofascore_id") is None:
                        continue
                    other = next(t for t in teams if t is not side)
                    games.setdefault(side["sofascore_id"], []).append({
                        "timestamp": record["timestamp"],
                        "scored": side["scored"],
                        "conceded": side["conceded"],
                        **team_summary(side["players"], other["players"]),
                    })
            for team_id, rows in load_cache(FORM_SEED_PATH).items():
                games.setdefault(int(team_id), []).extend(rows)
            for g in games.values():
                g.sort(key=lambda x: x["timestamp"], reverse=True)
            self._af_games = games
        return self._af_games

    def _latest_ranking(self, team_id: int, before_ts: int):
        """Most recent FIFA ranking for the team before the date, from the cached
        Sofascore games (no requests). Stopgap until a live ranking source is added."""
        best = None
        for e in self.events_cache.get(f"{team_id}", []):
            if e.get("startTimestamp", 0) >= before_ts:
                continue
            side = "home" if e.get("homeTeamId") == team_id else "away"
            rank = e.get(f"{side}Ranking")
            if rank is not None and (best is None or e["startTimestamp"] > best[0]):
                best = (e["startTimestamp"], rank)
        return best[1] if best else None

    def _apifootball_form_stats(self, team_id: int, team_name: str, before_ts: int) -> dict:
        """Same output as _last_n_form_stats, from the daily-collected API-Football games."""
        games = [g for g in self._apifootball_games().get(team_id, [])
                 if g["timestamp"] < before_ts and g["rating"] is not None][:N_MATCHES]
        if len(games) < N_MATCHES:
            raise Exception(f"Couldn't find {N_MATCHES} games with form stats for {team_name} "
                            f"(have {len(games)})")
        # Games the team played (with stats, per the Sofascore games cache) that are newer
        # than the oldest one used but missing from our history: too many means stale form.
        used = {g["timestamp"] for g in games}
        missing = [e for e in self.events_cache.get(f"{team_id}", [])
                   if e.get("hasStats") and games[-1]["timestamp"] < e.get("startTimestamp", 0) < before_ts
                   and e["startTimestamp"] not in used]
        if len(missing) > MAX_MISSING_FORM_GAMES:
            raise Exception(f"Form history for {team_name} is stale: {len(missing)} newer games are missing")
        ranking = self._latest_ranking(team_id, before_ts)
        if ranking is None:
            raise Exception(f"No FIFA ranking known for {team_name}")
        out = {"ranking": ranking}
        for i, g in enumerate(games, 1):
            out[f"fix_{i}"] = g["rating"]
            out[f"shots_{i}"] = g["shots"]
            out[f"scored_{i}"] = g["scored"]
            out[f"shots_against_{i}"] = g["shots_against"]
            out[f"relative_shots_{i}"] = _safe_ratio(g["shots"], g["shots_against"])
            out[f"relative_goals_{i}"] = _safe_ratio(g["scored"], g["conceded"])
        return out

    def warm_form_cache(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Refresh and compute Sofascore form stats for every row of df into the ratings
        cache, then trim the events cache of the teams involved. Returns the rows whose
        stats could be computed, with df's original columns.
        """
        if FORM_SOURCE != "sofascore":
            # API-Football stats are collected daily (src/collect_apifootball.py), so this
            # only computes them locally and drops matches that don't have them.
            return self._add_form_features(df)[df.columns.to_list()]
        out = self._add_sofascore_features(df)
        self._trim_events_cache(self._refreshed)
        save_cache(RATINGS_CACHE_PATH, self.ratings_cache)
        return out[df.columns.to_list()]

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
            # 'result' is carried through the pipeline so labels stay aligned with
            # the rows kept here (rows can be dropped below).
            X = df.copy()
            X = self._add_location_features(X)
            X = self._add_weather_features(X)
            X = self._add_form_features(X)
            if X.empty:
                print("no matches left after Sofascore features")
                sys.exit(1)
            X = self._add_derived_features(X)
            X.to_csv(f"{save_dir}/extracted.csv", index=False)
            return f"{save_dir}/extracted.csv"
        except FileNotFoundError as e:
            print(f"new matches are not found: {path}")
            sys.exit(1)
        except Exception as e:
            raise

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
