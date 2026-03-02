from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}

def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

class FolderDataset(Dataset):
    def __init__(self, root: str, classes: List[str], img_size: int, train: bool = True, seed: int = 0):
        self.root = Path(root)
        self.classes = classes
        self.img_size = img_size
        self.train = train
        self.rng = random.Random(seed)

        self.samples: List[Tuple[str, Path]] = []
        for c in classes:
            cdir = self.root / c
            if not cdir.exists():
                continue
            for p in list_images(cdir):
                self.samples.append((c, p))
        self.class_to_idx = {c:i for i,c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: Image.Image) -> Image.Image:
        if not self.train:
            return img
        angle = self.rng.uniform(-30, 30)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0))
        w,h = img.size
        m = min(w,h)
        left = max(0, (w-m)//2); top = max(0, (h-m)//2)
        img = img.crop((left, top, left+m, top+m))
        if self.rng.random() < 0.5:
            arr = np.array(img).astype(np.float32)
            arr = arr * self.rng.uniform(0.9, 1.1)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        c, path = self.samples[idx]
        img = Image.open(path)
        img = self._augment(img)
        x = self._to_tensor(img)
        y = self.class_to_idx[c]
        return x, y
