from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FILES = ("student_encoder.onnx", "gallery.npz", "student_encoder.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--models", type=Path, default=Path("artifacts/candidate"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/releases"))
    args = parser.parse_args()
    destination = args.output_root / f"student-recognition-v{args.version}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    checksums = []
    for name in FILES:
        source = args.models / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / name
        shutil.copy2(source, target)
        checksums.append(f"{sha256(target)}  {name}")
    (destination / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
    manifest = {"version": args.version, "files": list(FILES)}
    (destination / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(destination), "zip", root_dir=destination)
    print(archive)
