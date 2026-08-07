from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from .catalog import StudentCatalog
from .config import load_roster
from .geometry import FixedLessonLayout
from .recognition import OpenCVStudentRecognizer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_fixture(
    dataset_root: Path,
    model_dir: Path,
    fixture_name: str,
) -> dict:
    annotation_path = dataset_root / "annotations" / f"independent_test_annotations_{fixture_name}.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    fixture_dir = dataset_root / "lesson" / f"independent_{fixture_name}"
    catalog = StudentCatalog(load_roster())
    recognizer = OpenCVStudentRecognizer(model_dir, catalog)
    layout = FixedLessonLayout()
    rows = []
    identity_correct = 0
    eligibility_correct = 0
    wrong_clicks = 0
    for image_row in annotation["images"]:
        image_path = fixture_dir / image_row["file"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        selected_cards = set(image_row.get("selected_card_indices", []))
        for instance in image_row["instances"]:
            x1, y1, x2, y2 = instance["bbox"]
            prediction = recognizer.identify(image[y1:y2, x1:x2])
            predicted_eligible = layout.affection_eligible(image[y1:y2, x1:x2])
            card_index = int(instance["location"].split(":", 1)[0])
            selected = card_index in selected_cards or instance.get("card_state") == "selected"
            identity_ok = prediction.name == instance["name"]
            eligibility_ok = predicted_eligible == bool(instance["eligible"])
            would_click = prediction.name == instance["name"] and predicted_eligible and not selected
            expected_click = bool(instance["eligible"]) and not selected
            wrong_click = would_click and not expected_click
            identity_correct += int(identity_ok)
            eligibility_correct += int(eligibility_ok)
            wrong_clicks += int(wrong_click)
            rows.append(
                {
                    "file": image_row["file"],
                    "location": instance["location"],
                    "expected": instance["name"],
                    "top1": prediction.name,
                    "score": prediction.score,
                    "margin": prediction.margin,
                    "expected_eligible": bool(instance["eligible"]),
                    "predicted_eligible": predicted_eligible,
                    "card_state": "selected" if selected else "available",
                    "identity_correct": identity_ok,
                    "eligibility_correct": eligibility_ok,
                    "would_click_expected_target": would_click,
                    "wrong_click": wrong_click,
                }
            )
    return {
        "fixture": fixture_name,
        "annotation_sha256": _sha256(annotation_path),
        "model_sha256": _sha256(model_dir / "student_encoder.onnx"),
        "gallery_sha256": _sha256(model_dir / "gallery.npz"),
        "identity": {"correct": identity_correct, "total": len(rows)},
        "eligibility": {"correct": eligibility_correct, "total": len(rows)},
        "wrong_clicks": wrong_clicks,
        "instances": rows,
    }
