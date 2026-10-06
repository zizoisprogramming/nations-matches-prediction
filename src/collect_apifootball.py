"""
Daily collector of national-team match stats from API-Football's free plan.

The free plan only serves the current season for dates from yesterday to tomorrow,
so each team's recent-games history is built up here one day at a time. For every
finished senior national-team match of the day it stores, per team, the goals and
compact per-player rows (minutes, rating, shots, substitute flag); team_summary()
turns them into the measures the Sofascore form features use: average rating of
the starting XI, shots by the starters and by the opponent's starters.

    uv run python -m src.collect_apifootball            # yesterday (UTC)
    uv run python -m src.collect_apifootball 2026-10-05
"""
import argparse
import datetime as dt
import json
import re

from src.feature_extraction import _normalize_team_name
from src.helpers.apifootball import get, requests_left_today, ApiFootballError
from src.helpers.cache import load_cache, save_cache
from src.helpers.constants import APIFOOTBALL_MATCHES_PATH, CACHE_DIR

FINISHED = {"FT", "AET", "PEN"}
NON_SENIOR = re.compile(r"\bU\d\d\b|\bW$|Women|Olympic|Beach|Futsal|Clubs", re.IGNORECASE)
KEEP_SPARE_REQUESTS = 3


def team_index() -> dict:
    """Normalized team name (and its sorted words) -> Sofascore id, from team_ids.json."""
    index = {}
    for name, sofa_id in load_cache(CACHE_DIR / "team_ids.json").items():
        norm = _normalize_team_name(name)
        index[norm] = sofa_id
        index.setdefault(" ".join(sorted(norm.split())), sofa_id)
    return index


def match_team(name: str, index: dict):
    norm = _normalize_team_name(name)
    return index.get(norm) or index.get(" ".join(sorted(norm.split())))


def is_senior_national(fixture: dict, index: dict) -> bool:
    league, home, away = fixture["league"], fixture["teams"]["home"]["name"], fixture["teams"]["away"]["name"]
    if any(NON_SENIOR.search(s) for s in (league["name"], home, away)):
        return False
    known = (match_team(home, index) is not None) + (match_team(away, index) is not None)
    # Club competitions (e.g. the Champions League) are also listed under country
    # "World", so a "World" match needs at least one recognised national team.
    return known == 2 or (known == 1 and league.get("country") == "World")


def compact_players(players: list, team_id: int) -> list:
    """Per player: [minutes, rating, shots, substitute flag]. Small enough to keep, so the
    summary below can be recomputed later without spending API requests."""
    team = next((t for t in players if t["team"]["id"] == team_id), None)
    out = []
    for p in (team or {}).get("players", []):
        if not p["statistics"]:
            continue
        s = p["statistics"][0]
        rating = s["games"].get("rating")
        out.append([s["games"].get("minutes") or 0,
                    float(rating) if rating not in (None, "", "0") else None,
                    (s.get("shots") or {}).get("total") or 0,
                    bool(s["games"].get("substitute"))])
    return out


def starters(players: list) -> list:
    """The starting XI from compact player rows. API-Football's substitute flag is
    sometimes wrong (every squad member marked as a starter), so it's only trusted when
    it gives a believable XI; otherwise take the 11 players with the most minutes."""
    played = [p for p in players if p[0] > 0]
    flagged = [p for p in played if not p[3]]
    if 9 <= len(flagged) <= 11:
        return flagged
    return sorted(played, key=lambda p: p[0], reverse=True)[:11]


def team_summary(mine: list, theirs: list) -> dict:
    """Starting-XI average rating and shots, as the Sofascore features measure them."""
    xi, opp_xi = starters(mine), starters(theirs)
    ratings = [p[1] for p in xi if p[1]]
    return {
        "rating": round(sum(ratings) / len(ratings), 3) if len(ratings) >= 7 else None,
        "shots": sum(p[2] for p in xi),
        "shots_against": sum(p[2] for p in opp_xi),
    }


def collect(date: str) -> int:
    store = load_cache(APIFOOTBALL_MATCHES_PATH)
    index = team_index()
    fixtures = get("fixtures", date=date)
    todo = [f for f in fixtures
            if f["fixture"]["status"]["short"] in FINISHED
            and is_senior_national(f, index)
            and str(f["fixture"]["id"]) not in store]
    print(f"{date}: {len(fixtures)} fixtures, {len(todo)} new finished senior national-team matches")

    budget = requests_left_today() - KEEP_SPARE_REQUESTS
    added, unmatched = 0, set()
    for f in todo:
        if budget <= 0:
            print(f"  ⚠️  daily request budget used up; {len(todo) - added} matches not collected")
            break
        fid, home, away = f["fixture"]["id"], f["teams"]["home"], f["teams"]["away"]
        try:
            players = get("fixtures/players", fixture=fid)
        except ApiFootballError as e:
            print(f"  ✗ {home['name']} v {away['name']}: {e}")
            continue
        finally:
            budget -= 1
        gh, ga = f["goals"]["home"], f["goals"]["away"]
        record = {
            "date": f["fixture"]["date"][:10],
            "timestamp": f["fixture"]["timestamp"],
            "league": f["league"]["name"],
            "teams": {},
        }
        for side, scored, conceded in [(home, gh, ga), (away, ga, gh)]:
            sofa_id = match_team(side["name"], index)
            if sofa_id is None:
                unmatched.add(side["name"])
            record["teams"][str(side["id"])] = {
                "name": side["name"],
                "sofascore_id": sofa_id,
                "home": side is home,
                "scored": scored,
                "conceded": conceded,
                "players": compact_players(players, side["id"]),
            }
        store[str(fid)] = record
        added += 1
        hp, ap = record["teams"][str(home["id"])]["players"], record["teams"][str(away["id"])]["players"]
        h, a = team_summary(hp, ap), team_summary(ap, hp)
        print(f"  ✓ {home['name']} {gh}-{ga} {away['name']} ({record['league']}): "
              f"XI ratings {h['rating']} / {a['rating']}, XI shots {h['shots']} / {a['shots']}")
        save_cache(APIFOOTBALL_MATCHES_PATH, store)

    if unmatched:
        print(f"  ℹ️  not in team_ids.json (stored, but not used until an alias is added): {sorted(unmatched)}")
    print(f"collected {added} matches; {len(store)} stored in total")
    return added


def main():
    parser = argparse.ArgumentParser(description="Collect national-team match stats from API-Football")
    parser.add_argument("date", nargs="?", help="YYYY-MM-DD, default yesterday (UTC); free plan: yesterday to tomorrow")
    args = parser.parse_args()
    date = args.date or (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
    collect(date)


if __name__ == "__main__":
    main()
