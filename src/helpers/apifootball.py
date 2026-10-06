"""
Minimal API-Football (api-sports.io) client for the free plan: 100 requests/day,
10/minute, and only dates from yesterday to tomorrow for the current season.
The key comes from the API_FOOTBALL_KEY environment variable or the project's .env.
"""
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://v3.football.api-sports.io"
MIN_INTERVAL_SECONDS = 6.5          # stay under 10 requests/minute
_last_request_at = 0.0


class ApiFootballError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        env = Path(__file__).parent.parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("API_FOOTBALL_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise ApiFootballError("API_FOOTBALL_KEY is not set (environment or .env)")
    return key


def get(path: str, **params) -> list:
    """GET an endpoint and return its "response" list. Raises on API errors (e.g. plan limits)."""
    global _last_request_at
    wait = MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()
    r = requests.get(f"{BASE_URL}/{path}", params=params,
                     headers={"x-apisports-key": _api_key()}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise ApiFootballError(f"{path} {params}: {data['errors']}")
    return data.get("response", [])


def requests_left_today() -> int:
    status = get("status")
    req = status["requests"] if isinstance(status, dict) else status[0]["requests"]
    return req["limit_day"] - req["current"]
