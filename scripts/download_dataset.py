from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baas_student_recognition_ai.data import download_dataset, validate_dataset


if __name__ == "__main__":
    root, revision = download_dataset()
    manifest = validate_dataset(root)
    print(f"Downloaded dataset revision {revision} to {root}")
    print(manifest["counts"])
