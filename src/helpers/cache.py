
from pathlib import Path
import json

def load_cache(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)



