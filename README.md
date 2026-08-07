# BAAS Student Recognition AI

Training, evaluation, and release tooling for the lesson student identity
model used by [Blue Archive Auto Script](https://github.com/pur1fying/blue_archive_auto_script).

This repository intentionally contains **no training or evaluation images**.
The private dataset is downloaded from Hugging Face after the repository owner
authenticates locally with `hf auth login`.  The BAAS runtime never accesses
Hugging Face and never needs a Hugging Face token.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

Download the pinned private dataset:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

Train and export the MobileNetV3-Small encoder and prototype gallery:

```powershell
.\.venv\Scripts\python.exe scripts\train_identity.py
```

Build a versioned release archive after validation:

```powershell
.\.venv\Scripts\python.exe scripts\package_release.py --version 1.0.0
```

Release archives contain only the ONNX encoder, prototype gallery, metadata,
and SHA-256 checksums.  Training data, cached datasets, checkpoints, and full
prediction reports remain private and are excluded by `.gitignore`.

## Data policy

The private dataset contains user-supplied lesson screenshots, Git-history
portraits, roster montages, Wikiru portraits, annotations, and frozen
regression fixtures.  Source URLs and copyright notices remain part of the
dataset manifest.  Keeping a dataset private does not grant redistribution
rights to third-party assets.

## Runtime architecture

BAAS uses deterministic 1280x720-relative card/avatar geometry and simple
OpenCV frame analysis.  MobileNetV3-Small is used only for student identity.
YOLO is not part of the runtime or this training pipeline.

