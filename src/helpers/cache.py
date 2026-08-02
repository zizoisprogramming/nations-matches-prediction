
from pathlib import Path
import json

def _load_cache(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)



