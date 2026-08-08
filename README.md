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
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu124.txt
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

The CUDA 12.4 requirements are the default for GPU training. For a CPU-only
environment, install `requirements-torch-cpu.txt` instead. Keeping the PyTorch
wheel source separate prevents a normal PyPI install from silently replacing
the CUDA build with a CPU build.

Download the pinned private dataset:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

The repository owner first builds and uploads the private snapshot locally:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_hf_dataset.py --baas-root D:\BlueArchiveAutoScript_v1.4.3_win_x86_64_full_env --output .\dataset
.\.venv\Scripts\hf.exe upload kosetsu905/baas-student-recognition-data .\dataset . --repo-type dataset --commit-message "Upload student recognition dataset v1"
.\.venv\Scripts\python.exe scripts\pin_dataset_revision.py
```

Commit the resulting pinned `configs/dataset.json` revision before training.
Never commit `dataset/` or an authentication token.

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
