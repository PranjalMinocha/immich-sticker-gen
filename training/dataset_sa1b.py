"""
SA-1B / MobileSAM layout: paired .jpg and teacher .npy per image.
Deterministic 70% / 10% / 20% train / val / test split.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset


def collect_jpg_paths(root: Path, shard_dirs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for d in shard_dirs:
        folder = root / d
        if not folder.is_dir():
            raise FileNotFoundError(f"Data shard not found: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() == ".jpg":
                npy = p.with_suffix(".npy")
                if npy.is_file():
                    paths.append(p)
    if not paths:
        raise RuntimeError(f"No .jpg with paired .npy found under {root} / {shard_dirs}")
    return paths


def split_paths(
    paths: List[Path],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> Tuple[List[Path], List[Path], List[Path]]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Split produced empty partition (n={n}, train={n_train}, val={n_val}, test={n_test}). "
            "Add more images or adjust fractions."
        )
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def save_split_manifest(
    out_path: Path,
    train: List[Path],
    val: List[Path],
    test: List[Path],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "train": [str(p) for p in train],
        "val": [str(p) for p in val],
        "test": [str(p) for p in test],
    }
    out_path.write_text(json.dumps(payload, indent=2))


def load_split_manifest(path: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    data = json.loads(path.read_text())
    train = [Path(p) for p in data["train"]]
    val = [Path(p) for p in data["val"]]
    test = [Path(p) for p in data["test"]]
    return train, val, test


class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        if scale != 1.0:
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return image


def preprocess_image(rgb: np.ndarray, img_size: int) -> Tensor:
    x = ResizeLongestSide(img_size).apply_image(rgb)
    x = torch.as_tensor(x, dtype=torch.float32)
    x = x.permute(2, 0, 1).contiguous()
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    pad_h = img_size - h
    pad_w = img_size - w
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


class SA1BEncoderDataset(Dataset):
    def __init__(self, jpg_paths: Sequence[Path | str], img_size: int = 1024) -> None:
        self.paths = [Path(p) for p in jpg_paths]
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        jpg = self.paths[index]
        img = cv2.imread(str(jpg))
        if img is None:
            raise FileNotFoundError(str(jpg))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = preprocess_image(img, self.img_size)
        feat = np.load(str(jpg.with_suffix(".npy")))
        feat = np.squeeze(feat)
        if feat.ndim != 3:
            raise ValueError(f"Expected teacher .npy (C,H,W) after squeeze, got {feat.shape} for {jpg}")
        t = torch.from_numpy(feat.astype(np.float32))
        return x, t, str(jpg)


def subset_from_paths(full_dataset: SA1BEncoderDataset, paths: Sequence[Path]) -> Subset:
    path_set = {p.resolve() for p in paths}
    indices = [i for i, p in enumerate(full_dataset.paths) if p.resolve() in path_set]
    if len(indices) != len(paths):
        raise RuntimeError(
            f"subset_from_paths: expected {len(paths)} matches, got {len(indices)}"
        )
    return Subset(full_dataset, indices)


def build_datasets(
    data_root: Path,
    shard_dirs: Sequence[str],
    img_size: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_manifest: Path | None,
    split_manifest_out: Path | None,
):
    all_paths = collect_jpg_paths(data_root, shard_dirs)
    full = SA1BEncoderDataset(all_paths, img_size=img_size)

    if split_manifest is not None and split_manifest.is_file():
        train_p, val_p, test_p = load_split_manifest(split_manifest)
        meta = {"source": "manifest", "path": str(split_manifest)}
    else:
        train_p, val_p, test_p = split_paths(
            all_paths, seed, train_frac, val_frac, test_frac
        )
        meta = {
            "source": "fresh_split",
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
        }
        if split_manifest_out is not None:
            save_split_manifest(
                split_manifest_out,
                train_p,
                val_p,
                test_p,
                seed,
                train_frac,
                val_frac,
                test_frac,
            )
            meta["written_manifest"] = str(split_manifest_out)

    train_ds = subset_from_paths(full, train_p)
    val_ds = subset_from_paths(full, val_p)
    test_ds = subset_from_paths(full, test_p)
    meta["counts"] = {"train": len(train_p), "val": len(val_p), "test": len(test_p)}
    return full, train_ds, val_ds, test_ds, meta
