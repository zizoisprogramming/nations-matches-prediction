"""
Predict a single match: find it on ESPN for the venue, run the feature pipeline and
the current model.

    uv run python -m src.predict "Norway" "Portugal" 2026-09-27
"""
import argparse
import contextlib
import datetime as dt
import io
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.feature_extraction import FeatureExtraction, _normalize_team_name
from src.feature_scaling import FeatureScaling
from src.feature_selection import FeatureSelection
from src.inference import predict_proba
from src.scrape import scrape_espn, is_finished


def _same_team(a: str, b: str) -> bool:
    a, b = _normalize_team_name(a), _normalize_team_name(b)
    return a == b or set(a.split()) == set(b.split()) or SequenceMatcher(None, a, b).ratio() >= 0.85


def find_espn_fixture(team_a: str, team_b: str, date: str):
    """The ESPN fixture between the two teams on that date (either order), or None."""
    for m in scrape_espn(date) or []:
        if _same_team(team_a, m["home_team"]) and _same_team(team_b, m["away_team"]):
            return m
        if _same_team(team_b, m["home_team"]) and _same_team(team_a, m["away_team"]):
            return m
    return None


def predict_match(team_a: str, team_b: str, date: str, log=print) -> dict:
    """
    Predict team_a vs team_b on date (YYYY-MM-DD). If ESPN lists the match, its home
    team and venue are used; otherwise team_a is treated as home, playing in its capital.
    Returns a dict with the probabilities, the venue used and notes.
    """
    notes = []
    log(f"Looking up {team_a} vs {team_b} on ESPN for {date} ...")
    fixture = find_espn_fixture(team_a, team_b, date)

    if fixture:
        home_is_a = _same_team(team_a, fixture["home_team"])
        home, away = (team_a, team_b) if home_is_a else (team_b, team_a)
        city, country = fixture["city"], fixture["country"]
        tournament = fixture.get("tournament", "")
        if not home_is_a:
            notes.append(f"ESPN lists {home} as the home team, so the teams were swapped.")
        if not city or not country:
            notes.append("ESPN has no full venue for this match; using the home team's country.")
            city, country = city or "", country or home
    else:
        home, away, tournament = team_a, team_b, ""
        city, country = "", home
        notes.append(f"Match not found on ESPN for {date}; assuming {home} at home.")

    if not city:
        fe = FeatureExtraction()
        capital = fe._get_capital(home, fe.capitals_cache)
        city = capital or country
    d = dt.date.fromisoformat(date)
    row = dict(date=date, home_team=home, away_team=away, tournament=tournament,
               city=city, country=country, year=d.year, month=d.month, day=d.day)

    log(f"Building features for {home} vs {away} at {city}, {country} (Sofascore form, weather, travel) ...")
    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame([row]).to_csv(f"{tmp}/input.csv", index=False)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                FeatureExtraction().run(f"{tmp}/input.csv", tmp)
                FeatureSelection().run(f"{tmp}/extracted.csv", tmp)
                FeatureScaling().run(f"{tmp}/selected.csv", tmp)
        except SystemExit:
            pass
        if not Path(f"{tmp}/scaled.csv").exists() or pd.read_csv(f"{tmp}/scaled.csv").empty:
            reasons = [l for l in out.getvalue().splitlines()
                       if "Couldn't" in l or "Dropping" in l or "blocking" in l or "no matches" in l]
            raise RuntimeError("Couldn't build features for this match. " + " | ".join(reasons[-3:]))
        probs = predict_proba(f"{tmp}/scaled.csv").iloc[0]

    result = {
        "home_team": home, "away_team": away, "date": date,
        "city": city, "country": country, "tournament": tournament,
        "prob_home_win": float(probs["prob_home_win"]),
        "prob_draw": float(probs["prob_draw"]),
        "prob_away_win": float(probs["prob_away_win"]),
        "notes": notes,
        "actual": None,
    }
    if fixture and is_finished(fixture) and fixture.get("home_score") is not None:
        result["actual"] = f"{fixture['home_team']} {fixture['home_score']} - {fixture['away_score']} {fixture['away_team']}"
    elif fixture and fixture.get("status"):
        notes.append(f"Match status on ESPN: {fixture['status']!r}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict one match")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("date", help="YYYY-MM-DD")
    args = parser.parse_args()
    r = predict_match(args.team_a, args.team_b, args.date)
    print(f"\n{r['home_team']} vs {r['away_team']} ({r['date']}, {r['city']}, {r['country']})")
    print(f"  home win {r['prob_home_win']:.0%} | draw {r['prob_draw']:.0%} | away win {r['prob_away_win']:.0%}")
    for n in r["notes"]:
        print(f"  note: {n}")
    if r["actual"]:
        print(f"  actual result: {r['actual']}")


if __name__ == "__main__":
    main()
