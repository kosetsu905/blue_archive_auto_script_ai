from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Copy checksum mismatch: {source}")


def copy_directory(source: Path, destination: Path) -> None:
    for path in sorted(source.iterdir()):
        if path.is_file():
            copy_file(path, destination / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the private Hugging Face dataset upload tree")
    parser.add_argument("--baas-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset-revision", default="local-unpublished")
    args = parser.parse_args()
    baas_root = args.baas_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output}")
    source_data = baas_root / "develop_tools" / "student_recognition" / "data"
    source_tools = baas_root / "develop_tools" / "student_recognition"
    source_fixtures = baas_root / "develop_tools" / "test" / "fixtures"

    copy_directory(source_data / "historical_portraits", output / "portraits" / "historical")
    copy_directory(source_data / "wikiru_portraits", output / "portraits" / "wikiru")
    copy_directory(source_data / "roster_montages", output / "roster_montages")
    copy_directory(source_fixtures / "lesson", output / "lesson" / "train")
    copy_directory(source_fixtures / "lesson_independent_v1", output / "lesson" / "independent_v1")
    copy_directory(source_fixtures / "lesson_independent_v2", output / "lesson" / "independent_v2")

    annotations = (
        "lesson_locator_annotations.json",
        "independent_test_annotations_v1.json",
        "independent_test_annotations_v2.json",
        "historical_label_corrections.json",
    )
    for name in annotations:
        copy_file(source_tools / name, output / "annotations" / name)

    history_rows = json.loads((output / "portraits" / "historical" / "manifest.json").read_text(encoding="utf-8"))
    wikiru = json.loads((output / "portraits" / "wikiru" / "manifest.json").read_text(encoding="utf-8"))
    montage = json.loads((output / "roster_montages" / "roster_montage_annotations.json").read_text(encoding="utf-8"))
    train = json.loads((output / "annotations" / "lesson_locator_annotations.json").read_text(encoding="utf-8"))
    v1 = json.loads((output / "annotations" / "independent_test_annotations_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((output / "annotations" / "independent_test_annotations_v2.json").read_text(encoding="utf-8"))
    counts = {
        "historical_portraits": len(history_rows),
        "wikiru_portraits": len(wikiru["entries"]),
        "wikiru_training_portraits": sum(bool(row["include_for_identity_training"]) for row in wikiru["entries"]),
        "roster_montages": len(montage["files"]),
        "lesson_training_screenshots": len(train["images"]),
        "independent_v1_screenshots": len(v1["images"]),
        "independent_v2_screenshots": len(v2["images"]),
    }
    expected = {
        "historical_portraits": 177,
        "wikiru_portraits": 272,
        "wikiru_training_portraits": 270,
        "roster_montages": 9,
        "lesson_training_screenshots": 5,
        "independent_v1_screenshots": 5,
        "independent_v2_screenshots": 11,
    }
    if counts != expected:
        raise ValueError(f"Dataset count mismatch: {counts}")
    source_notice = """# Dataset source and access notice\n\nThis private dataset is maintained for BAAS student-recognition research. It contains user-provided screenshots and third-party game artwork. Copyright remains with the respective rights holders. Private storage does not grant redistribution rights. See each source manifest for URLs, capture dates, hashes, and inclusion decisions. Frozen `independent_v1` and `independent_v2` fixtures must never be enumerated by the training loader.\n"""
    (output / "README.md").write_text(source_notice, encoding="utf-8")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "version": 1,
        "dataset_revision": args.dataset_revision,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "split_policy": {
            "training": ["portraits/historical", "portraits/wikiru", "roster_montages", "lesson/train"],
            "frozen_evaluation_only": ["lesson/independent_v1", "lesson/independent_v2"],
        },
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "counts": counts, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
