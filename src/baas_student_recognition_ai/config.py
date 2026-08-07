from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_roster() -> list[dict[str, str]]:
    payload = load_json(ROOT / "configs" / "roster.json")
    rows = payload["student_names"]
    if len(rows) != 270 or len({row["Global_name"] for row in rows}) != 270:
        raise ValueError("The training roster must contain 270 unique identities")
    expected = {"CN_name", "Global_name", "JP_name"}
    if any(set(row) != expected for row in rows):
        raise ValueError("Roster entries must contain only the three aliases")
    return rows


def load_dataset_config() -> dict[str, Any]:
    return load_json(ROOT / "configs" / "dataset.json")


def load_training_config() -> dict[str, Any]:
    return load_json(ROOT / "configs" / "training.json")

