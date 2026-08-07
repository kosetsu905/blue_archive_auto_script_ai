from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .catalog import StudentCatalog
from .training import normalized_student_view


@dataclass(frozen=True)
class Prediction:
    name: str
    score: float
    margin: float


class OpenCVStudentRecognizer:
    """Runtime-equivalent global Top-1 inference used by private evaluation."""

    def __init__(self, model_dir: Path, catalog: StudentCatalog) -> None:
        metadata = json.loads(
            (model_dir / "student_encoder.json").read_text(encoding="utf-8")
        )
        gallery = np.load(model_dir / "gallery.npz", allow_pickle=False)
        embeddings = np.asarray(gallery["embeddings"], dtype=np.float32)
        ids = np.asarray(gallery["student_ids"]).astype(str)
        width = int(metadata["embedding_size"])
        if embeddings.ndim != 2 or embeddings.shape[1] != width or len(ids) != len(embeddings):
            raise ValueError("Invalid gallery dimensions")
        self.net = cv2.dnn.readNetFromONNX(str(model_dir / "student_encoder.onnx"))
        self.embeddings = embeddings
        self.ids = ids
        self.catalog = catalog

    def identify(self, crop: np.ndarray) -> Prediction:
        import random

        view = normalized_student_view(crop, False, random.Random(0))[None]
        self.net.setInput(view)
        embedding = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Encoder and gallery embedding widths differ")
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        scores = embedding @ self.embeddings.T
        best_by_id: dict[str, float] = {}
        for student_id, score in zip(self.ids, scores):
            best_by_id[student_id] = max(best_by_id.get(student_id, -1.0), float(score))
        ranking = sorted(best_by_id.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranking[0]
        margin = best_score - ranking[1][1] if len(ranking) > 1 else best_score
        return Prediction(self.catalog.records[best_id].canonical_name, best_score, margin)
