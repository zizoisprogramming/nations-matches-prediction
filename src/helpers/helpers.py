
import pandas as pd
from pathlib import Path


def _safe_ratio(a, b):
    if a is None or b is None:
        return None
    total = a + b
    return 0.5 if total == 0 else round(a / total, 4)



def slim_event(event: dict) -> dict:
    """Keep only the fields we need — discard everything else."""
    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})

    return {
        "id":             event.get("id"),
        "startTimestamp": event.get("startTimestamp"),
        "status":         event.get("status", {}).get("type"),
        "homeTeamId":     home.get("id"),
        "awayTeamId":     away.get("id"),
        "homeScore":      event.get("homeScore", {}).get("current"),
        "awayScore":      event.get("awayScore", {}).get("current"),
        "homeRanking":    home.get("ranking"),
        "awayRanking":    away.get("ranking"),
        "hasStats":       event.get("hasEventPlayerStatistics", False),
    }