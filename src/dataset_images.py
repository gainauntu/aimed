from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import random

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: str
    target: int


def scan_class_folders(root: str) -> Tuple[List[str], Dict[str, int], List[Sample]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    classes = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found under: {root}")

    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples: List[Sample] = []

    for cls_name in classes:
        cls_dir = root_path / cls_name
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append(Sample(path=str(p), target=class_to_idx[cls_name]))

    if not samples:
        raise RuntimeError(f"No image files found under: {root}")

    return classes, class_to_idx, samples


def split_samples_stratified(
    samples: Sequence[Sample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}
    for s in samples:
        by_class.setdefault(s.target, []).append(s)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    for _, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    if not train_samples:
        raise RuntimeError("Train split became empty. Check dataset size / val_ratio.")

    return train_samples, val_samples


def build_train_transform(img_size: int) -> Callable:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        T.RandomRotation(8),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform(img_size: int) -> Callable:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class PillImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Callable | None = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            img = Image.open(s.path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {s.path}") from e

        if self.transform is not None:
            img = self.transform(img)

        return img, s.target, s.path
