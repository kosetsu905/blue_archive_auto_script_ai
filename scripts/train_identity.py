from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baas_student_recognition_ai.data import download_dataset, validate_dataset
from baas_student_recognition_ai.training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate"))
    args = parser.parse_args()
    dataset = args.dataset
    dataset_revision = args.dataset_revision
    if dataset is None:
        dataset, dataset_revision = download_dataset()
    validate_dataset(dataset)
    report = train(dataset, args.output, dataset_revision=dataset_revision)
    print(report["opencv_training_replay"])
