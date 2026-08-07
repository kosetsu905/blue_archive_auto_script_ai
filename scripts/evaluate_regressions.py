from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baas_student_recognition_ai.evaluation import evaluate_fixture


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/private-regression-report.json"))
    args = parser.parse_args()
    payload = {
        "independent_v1": evaluate_fixture(args.dataset, args.models, "v1"),
        "independent_v2": evaluate_fixture(args.dataset, args.models, "v2"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
