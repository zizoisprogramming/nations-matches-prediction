"""
Seed the form-stats history from the Sofascore caches, so teams have their last 2
games before API-Football collection (src/collect_apifootball.py) has caught up.

For each team, the most recent ratings-cache entry with complete stats holds its 2
latest games with stats before that date (starting-XI rating, shots for/against,
goals). Those games are written to form_seed.json with their dates and scores from
the Sofascore games cache; newer API-Football games then take their place in the
"last 2 games" as they're collected.

    uv run python -m src.seed_form_cache
"""
from src.helpers.cache import load_cache, save_cache
from src.helpers.constants import CACHE_DIR, FORM_SEED_PATH

DAY = 86_400


def _conceded(scored, relative_goals):
    """Goals against from goals for and scored / (scored + conceded)."""
    if scored is None or relative_goals is None:
        return None
    if relative_goals == 0.5 and scored == 0:
        return 0
    if relative_goals == 0:
        return None
    return round(scored * (1 - relative_goals) / relative_goals)


def build_seed() -> dict:
    ratings = load_cache(CACHE_DIR / "ratings_cache.json")
    events = load_cache(CACHE_DIR / "team_events.json")

    latest = {}   # team id -> (date timestamp, entry)
    for key, entry in ratings.items():
        team_id, ts = key.rsplit("_", 1)
        fix = entry.get("fix") or []
        if len(fix) < 2 or any(v is None for v in fix[:2]) or entry.get("ranking") is None:
            continue
        if int(ts) > latest.get(team_id, (0, None))[0]:
            latest[team_id] = (int(ts), entry)

    seed = {}
    for team_id, (date_ts, entry) in latest.items():
        # The entry's games are the team's newest games with stats before date_ts.
        games = sorted((e for e in events.get(team_id, [])
                        if e.get("startTimestamp", 0) < date_ts and e.get("hasStats")),
                       key=lambda e: e["startTimestamp"], reverse=True)[:2]
        rows = []
        for i in range(2):
            event = games[i] if i < len(games) else None
            scored = entry["scored"][i]
            conceded = None
            if event is not None:
                home = event.get("homeTeamId") == int(team_id)
                conceded = event.get("awayScore" if home else "homeScore")
            if conceded is None:
                conceded = _conceded(scored, entry["relative_goals"][i])
            rows.append({
                "timestamp": event["startTimestamp"] if event else date_ts - (i + 1) * DAY,
                "rating": entry["fix"][i],
                "shots": entry["shots"][i],
                "shots_against": entry["shots_against"][i],
                "scored": scored,
                "conceded": conceded,
                "source": "sofascore",
            })
        seed[team_id] = rows
    return seed


def main():
    seed = build_seed()
    # Keep games added later by hand (e.g. a one-off backfill of missing games).
    for team_id, rows in load_cache(FORM_SEED_PATH).items():
        extra = [r for r in rows if r.get("source") != "sofascore"]
        seen = {r["timestamp"] for r in seed.get(team_id, [])}
        seed.setdefault(team_id, []).extend(r for r in extra if r["timestamp"] not in seen)
    save_cache(FORM_SEED_PATH, seed)
    print(f"seeded the last 2 games for {len(seed)} teams into {FORM_SEED_PATH.name}")


if __name__ == "__main__":
    main()
