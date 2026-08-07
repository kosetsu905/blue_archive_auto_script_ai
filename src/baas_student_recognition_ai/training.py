from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .catalog import StudentCatalog
from .config import ROOT, load_roster, load_training_config
from .data import Portrait, group_by_identity, load_training_portraits, validate_dataset
from .models import PretrainedMobileNetV3StudentEncoder, StudentEncoderTrainer


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def composite_alpha(image: np.ndarray, gray: int) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Portrait must be HWC")
    if image.shape[2] == 3:
        return image.copy()
    if image.shape[2] != 4:
        raise ValueError("Portrait must have BGR or BGRA channels")
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    background = np.full(image.shape[:2] + (3,), gray, dtype=np.float32)
    return np.clip(image[:, :, :3] * alpha + background * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )


def normalized_student_view(
    image: np.ndarray,
    randomize: bool,
    rng: random.Random,
    input_size: int = 96,
) -> np.ndarray:
    gray = rng.randint(205, 235) if randomize else 220
    source = composite_alpha(image, gray)
    source_height, source_width = source.shape[:2]
    extent = 90.0
    scale = extent / max(source_height, source_width)
    if randomize:
        scale *= rng.uniform(0.70, 1.40)
    target_width = max(8, round(source_width * scale))
    target_height = max(8, round(source_height * scale * (rng.uniform(0.92, 1.08) if randomize else 1.0)))
    interpolation = rng.choice(
        [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
    ) if randomize else cv2.INTER_AREA
    resized = cv2.resize(source, (target_width, target_height), interpolation=interpolation)
    canvas = np.full((input_size, input_size, 3), gray, dtype=np.uint8)
    max_x = input_size - target_width
    max_y = input_size - target_height
    x = max_x // 2 if not randomize else rng.randint(min(0, max_x), max(0, max_x))
    y = max_y // 2 if not randomize else rng.randint(min(0, max_y), max(0, max_y))
    src_x1, src_y1 = max(0, -x), max(0, -y)
    dst_x1, dst_y1 = max(0, x), max(0, y)
    copy_width = min(target_width - src_x1, input_size - dst_x1)
    copy_height = min(target_height - src_y1, input_size - dst_y1)
    if copy_width > 0 and copy_height > 0:
        canvas[dst_y1:dst_y1 + copy_height, dst_x1:dst_x1 + copy_width] = resized[
            src_y1:src_y1 + copy_height,
            src_x1:src_x1 + copy_width,
        ]
    if randomize:
        alpha = rng.uniform(0.80, 1.20)
        beta = rng.uniform(-18.0, 18.0)
        canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)
        if rng.random() < 0.25:
            canvas = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.2, 1.0))
        if rng.random() < 0.25:
            quality = rng.randint(55, 95)
            success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if success:
                canvas = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if rng.random() < 0.35:
            width = rng.randint(5, 18)
            height = rng.randint(5, 18)
            x1 = rng.randint(0, input_size - width)
            y1 = rng.randint(0, input_size - height)
            canvas[y1:y1 + height, x1:x1 + width] = rng.randint(170, 235)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    return normalized.transpose(2, 0, 1).astype(np.float32)


def _portrait_views(crop: np.ndarray) -> list[np.ndarray]:
    """Match the deterministic inner translations used by the BAAS runtime."""
    height, width = crop.shape[:2]
    x1 = min(3, max(0, width - 1))
    y1 = min(3, max(0, height - 1))
    x2 = max(x1 + 1, width - max(5, round(width * 0.12)))
    y2 = max(y1 + 1, height - max(5, round(height * 0.12)))
    inner = crop[y1:y2, x1:x2]
    inner_height, inner_width = inner.shape[:2]
    views = [inner]
    for ratio, x_ratio, y_ratio in (
        (0.86, 0.0, 0.0),
        (0.86, 1.0, 0.0),
        (0.78, 0.5, 0.0),
        (0.78, 0.5, 1.0),
    ):
        view_width = max(8, round(inner_width * ratio))
        view_height = max(8, round(inner_height * ratio))
        offset_x = round((inner_width - view_width) * x_ratio)
        offset_y = round((inner_height - view_height) * y_ratio)
        views.append(inner[offset_y:offset_y + view_height, offset_x:offset_x + view_width])
    return views


def runtime_student_views(crop: np.ndarray) -> np.ndarray:
    views = []
    for view in _portrait_views(crop):
        height, width = view.shape[:2]
        scale = min(96 / width, 96 / height)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(view, target, interpolation=interpolation)
        canvas = np.full((96, 96, 3), 220, dtype=np.uint8)
        x = (96 - target[0]) // 2
        y = (96 - target[1]) // 2
        canvas[y:y + target[1], x:x + target[0]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        views.append(((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))
    return np.stack(views)


class IdentityBalancedDataset(Dataset):
    def __init__(
        self,
        portraits: list[Portrait],
        label_to_index: dict[str, int],
        seed: int,
        minimum_draws: int,
    ) -> None:
        self.groups = group_by_identity(portraits)
        missing = set(label_to_index) - set(self.groups)
        if missing:
            raise ValueError(f"Identities without training images: {sorted(missing)}")
        self.names = sorted(label_to_index)
        self.label_to_index = label_to_index
        self.seed = seed
        self.epoch = 0
        self.draws = max(minimum_draws, max(len(rows) for rows in self.groups.values()))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.names) * self.draws

    def __getitem__(self, index: int):
        identity_index, draw = divmod(index, self.draws)
        name = self.names[identity_index]
        rows = list(self.groups[name])
        ordering = random.Random(self.seed + self.epoch * 1000003 + identity_index)
        ordering.shuffle(rows)
        portrait = rows[draw % len(rows)]
        first = normalized_student_view(
            portrait.image,
            True,
            random.Random(self.seed + self.epoch * 10000019 + index * 2),
        )
        second = normalized_student_view(
            portrait.image,
            True,
            random.Random(self.seed + self.epoch * 10000019 + index * 2 + 1),
        )
        return torch.from_numpy(np.stack([first, second])), self.label_to_index[name]


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * ~identity
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    return -(log_probability * positive).sum(dim=1)[valid].div(positive_count[valid]).mean()


def train_stage(
    portraits: list[Portrait],
    names: list[str],
    epochs: int,
    frozen_epochs: int,
    seed: int,
    config: dict,
    initial_state: dict | None = None,
) -> PretrainedMobileNetV3StudentEncoder:
    label_to_index = {name: index for index, name in enumerate(names)}
    dataset = IdentityBalancedDataset(
        portraits,
        label_to_index,
        seed,
        int(config["samples_per_identity"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PretrainedMobileNetV3StudentEncoder(int(config["embedding_size"]))
    if initial_state is not None:
        encoder.load_state_dict(initial_state)
    model = StudentEncoderTrainer(len(names), encoder=encoder).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.features.parameters(),
                "lr": float(config["learning_rate_backbone"]),
            },
            {
                "params": model.encoder.projection.parameters(),
                "lr": float(config["learning_rate_head"]),
            },
            {
                "params": model.classifier.parameters(),
                "lr": float(config["learning_rate_head"]),
            },
        ],
        weight_decay=1e-4,
    )
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        backbone_trainable = initial_state is not None or epoch >= frozen_epochs
        for parameter in model.encoder.features.parameters():
            parameter.requires_grad = backbone_trainable
        model.encoder.features.train(backbone_trainable)
        running = 0.0
        for views, labels in loader:
            views, labels = views.to(device), labels.to(device)
            images = views.reshape(-1, 3, 96, 96)
            expanded = labels.repeat_interleave(2)
            optimizer.zero_grad()
            embeddings, logits = model(images)
            loss = F.cross_entropy(logits, expanded)
            loss = loss + float(config["contrastive_weight"]) * supervised_contrastive_loss(
                embeddings, expanded
            )
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:03d} loss={running / max(1, len(loader)):.6f}")
    return model.encoder.to("cpu").eval()


def _batch_embeddings(
    encoder: nn.Module,
    portraits: list[Portrait],
    batch_size: int = 128,
) -> np.ndarray:
    inputs = [
        runtime_student_views(row.image)[0]
        if row.source.startswith("lesson:")
        else normalized_student_view(row.image, False, random.Random(0))
        for row in portraits
    ]
    output = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.stack(inputs[start:start + batch_size]))
            output.append(encoder(batch).numpy())
    return np.concatenate(output).astype(np.float32)


def build_gallery(
    encoder: nn.Module,
    portraits: list[Portrait],
    catalog: StudentCatalog,
) -> tuple[np.ndarray, np.ndarray, dict]:
    embeddings = _batch_embeddings(encoder, portraits)
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}
    for portrait, embedding in zip(portraits, embeddings):
        record = catalog.resolve(portrait.name)
        if record is None:
            raise ValueError(f"Unknown portrait identity: {portrait.name}")
        source_group = portrait.source.split(":", 1)[0]
        grouped.setdefault(record.student_id, {}).setdefault(source_group, []).append(embedding)
    gallery_embeddings = []
    gallery_ids = []
    support = {}
    priority = ("lesson", "wikiru", "roster", "history")
    for student_id in sorted(catalog.records):
        sources = grouped.get(student_id, {})
        if not sources:
            raise ValueError(f"No prototype for {student_id}")
        selected = []
        for source in priority:
            values = sources.get(source)
            if not values:
                continue
            centroid = np.mean(np.stack(values), axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            selected.append((source, centroid.astype(np.float32)))
            if len(selected) == 3:
                break
        for _, embedding in selected:
            gallery_embeddings.append(embedding)
            gallery_ids.append(student_id)
        support[student_id] = {
            "sources": sorted(sources),
            "prototype_sources": [source for source, _ in selected],
        }
    return np.stack(gallery_embeddings), np.asarray(gallery_ids), support


def export_onnx(encoder: nn.Module, path: Path, opset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        torch.zeros(1, 3, 96, 96),
        str(path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
    )


def opencv_replay(
    onnx_path: Path,
    portraits: list[Portrait],
    gallery_embeddings: np.ndarray,
    gallery_ids: np.ndarray,
    catalog: StudentCatalog,
) -> dict:
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    correct = 0
    failures = []
    started = time.perf_counter()
    for portrait in portraits:
        views = (
            runtime_student_views(portrait.image)
            if portrait.source.startswith("lesson:")
            else normalized_student_view(portrait.image, False, random.Random(0))[None]
        )
        net.setInput(views)
        embeddings = np.asarray(net.forward(), dtype=np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        similarities = np.max(embeddings @ gallery_embeddings.T, axis=0)
        best_by_id: dict[str, float] = {}
        for student_id, score in zip(gallery_ids, similarities):
            best_by_id[str(student_id)] = max(best_by_id.get(str(student_id), -1.0), float(score))
        predicted_id = max(best_by_id, key=best_by_id.get)
        expected = catalog.resolve(portrait.name)
        if expected is not None and predicted_id == expected.student_id:
            correct += 1
        else:
            failures.append(
                {
                    "source": portrait.source,
                    "expected": portrait.name,
                    "predicted_id": predicted_id,
                }
            )
    return {
        "correct": correct,
        "total": len(portraits),
        "failures": failures,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def train(
    dataset_root: Path,
    output_dir: Path,
    dataset_revision: str | None = None,
) -> dict:
    dataset_manifest = validate_dataset(dataset_root)
    revision = dataset_revision or str(dataset_manifest["dataset_revision"])
    config = load_training_config()
    seed = int(config["seed"])
    seed_everything(seed)
    catalog = StudentCatalog(load_roster())
    names = [catalog.records[student_id].canonical_name for student_id in sorted(catalog.records)]
    seeds, lesson = load_training_portraits(dataset_root)
    pretrain = train_stage(
        seeds,
        names,
        int(config["pretrain_epochs"]),
        int(config["frozen_backbone_epochs"]),
        seed,
        config,
    )
    encoder = train_stage(
        seeds + lesson,
        names,
        int(config["final_epochs"]),
        0,
        seed,
        config,
        initial_state=pretrain.state_dict(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "student_encoder.onnx"
    export_onnx(encoder, onnx_path, int(config["opset"]))
    gallery, gallery_ids, support = build_gallery(encoder, seeds + lesson, catalog)
    np.savez_compressed(output_dir / "gallery.npz", embeddings=gallery, student_ids=gallery_ids)
    metadata = {
        "version": 1,
        "architecture": "torchvision-MobileNetV3-Small-ImageNet1K-V1",
        "embedding_size": int(config["embedding_size"]),
        "input_width": int(config["input_size"]),
        "input_height": int(config["input_size"]),
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "identity_count": len(catalog.records),
        "prototype_count": len(gallery_ids),
        "identity_click_policy": "valid_global_top1",
        "dataset_revision": revision,
        "student_support": support,
    }
    (output_dir / "student_encoder.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    replay = opencv_replay(
        onnx_path,
        seeds + lesson,
        gallery,
        gallery_ids,
        catalog,
    )
    report = {
        "seed": seed,
        "dataset_revision": revision,
        "source_counts": {"seed": len(seeds), "lesson": len(lesson)},
        "opencv_training_replay": replay,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
