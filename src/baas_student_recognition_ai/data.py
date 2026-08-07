from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from huggingface_hub import snapshot_download

from .config import ROOT, load_dataset_config


@dataclass(frozen=True)
class Portrait:
    name: str
    source: str
    image: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_dataset() -> tuple[Path, str]:
    config = load_dataset_config()
    revision = str(config["revision"])
    path = Path(
        snapshot_download(
            repo_id=str(config["repo_id"]),
            repo_type=str(config.get("repo_type", "dataset")),
            revision=revision,
            local_dir=ROOT / str(config.get("local_dir", "dataset")),
            token=True,
        )
    )
    info = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return path, str(info["dataset_revision"])


def validate_dataset(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["files"]
    seen: set[str] = set()
    for row in rows:
        relative = str(row["path"])
        if relative in seen:
            raise ValueError(f"Duplicate dataset path: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {relative}")
    expected = payload["counts"]
    if expected != {
        "historical_portraits": 177,
        "wikiru_portraits": 272,
        "wikiru_training_portraits": 270,
        "roster_montages": 9,
        "lesson_training_screenshots": 5,
        "independent_v1_screenshots": 5,
        "independent_v2_screenshots": 11,
    }:
        raise ValueError(f"Unexpected dataset counts: {expected}")
    return payload


def _checked_image(path: Path, expected_sha256: str, unchanged: bool = False) -> np.ndarray:
    if sha256(path) != expected_sha256:
        raise ValueError(f"Portrait checksum mismatch: {path}")
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Cannot decode portrait: {path}")
    return image


def load_historical(root: Path) -> list[Portrait]:
    directory = root / "portraits" / "historical"
    rows = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return [
        Portrait(
            name=row["label"],
            source=f"history:{row['git_blob']}",
            image=_checked_image(directory / row["file"], row["sha256"]),
        )
        for row in rows
    ]


def load_wikiru(root: Path) -> list[Portrait]:
    directory = root / "portraits" / "wikiru"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    portraits = []
    for row in manifest["entries"]:
        image = _checked_image(directory / row["file"], row["sha256"], unchanged=True)
        if row["include_for_identity_training"]:
            portraits.append(
                Portrait(row["config_name"], f"wikiru:{row['sha256']}", image)
            )
    return portraits


def load_roster(root: Path) -> list[Portrait]:
    directory = root / "roster_montages"
    annotation = json.loads(
        (directory / "roster_montage_annotations.json").read_text(encoding="utf-8")
    )
    language = annotation["training_pixel_language"]
    images = {}
    for row in annotation["files"]:
        if row["language"] == language:
            images[row["image_index"]] = _checked_image(
                directory / row["file"], row["sha256"]
            )
    origin_x, origin_y = annotation["grid"]["origin"]
    step_x, step_y = annotation["grid"]["step"]
    width, height = annotation["grid"]["portrait_size"]
    portraits = []
    for row in annotation["entries"]:
        if not row["include_for_identity_training"]:
            continue
        x1 = origin_x + (row["column"] - 1) * step_x
        y1 = origin_y + (row["row"] - 1) * step_y
        crop = images[row["image_index"]][y1:y1 + height, x1:x1 + width].copy()
        portraits.append(
            Portrait(
                row["config_name"],
                f"roster:{language}:{row['image_index']}:{row['row']}:{row['column']}",
                cv2.resize(crop, (33, 30), interpolation=cv2.INTER_AREA),
            )
        )
    return portraits


def load_lesson_training(root: Path) -> list[Portrait]:
    directory = root / "lesson" / "train"
    annotation = json.loads(
        (root / "annotations" / "lesson_locator_annotations.json").read_text(
            encoding="utf-8"
        )
    )
    card_boxes = annotation["card_boxes"]
    geometry = annotation["avatar_geometry"]
    portraits = []
    for image_name, details in annotation["images"].items():
        screenshot = cv2.imread(str(directory / image_name))
        if screenshot is None:
            raise FileNotFoundError(directory / image_name)
        for location, name in details["identity_labels"].items():
            card_index, avatar_slot = (int(value) for value in location.split(":"))
            card_x, card_y = card_boxes[card_index][:2]
            x1 = card_x + geometry["relative_x"][avatar_slot]
            y1 = card_y + geometry["relative_y"]
            crop = screenshot[
                y1:y1 + geometry["height"],
                x1:x1 + geometry["width"],
            ].copy()
            portraits.append(Portrait(name, f"lesson:{image_name}:{location}", crop))
    return portraits


def load_training_portraits(root: Path) -> tuple[list[Portrait], list[Portrait]]:
    seeds = load_historical(root) + load_wikiru(root) + load_roster(root)
    lesson = load_lesson_training(root)
    return seeds, lesson


def group_by_identity(portraits: Iterable[Portrait]) -> dict[str, list[Portrait]]:
    groups: dict[str, list[Portrait]] = {}
    for portrait in portraits:
        groups.setdefault(portrait.name, []).append(portrait)
    return groups

