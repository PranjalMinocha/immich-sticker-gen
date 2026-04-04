"""
SA-1B-style data: encoder distillation (.jpg + teacher .npy) and optional mask JSON for full-SAM training.

Data paths: **data.data_dir** (JPEGs + optional mask JSON beside them) and **data.embeddings_dir**
(teacher **{stem}.npy** matching each **{stem}.jpg**). See training/DATA.md.

Training reads files via the filesystem (e.g. rclone mount). It does not load the full dataset into RAM;
throughput depends on mount latency. Extract tar archives to a directory first — see training/DATA.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset

try:
    from pycocotools import mask as mask_util
except ImportError:
    mask_util = None


def collect_encoder_pairs(data_cfg: dict) -> List[Tuple[Path, Path]]:
    if "data_dir" not in data_cfg or "embeddings_dir" not in data_cfg:
        if any(
            k in data_cfg
            for k in ("root", "shard_dirs", "image_root", "teacher_root", "layout")
        ):
            raise ValueError(
                "Training expects data.data_dir and data.embeddings_dir only. "
                "Remove layout, root, shard_dirs, image_root, and teacher_root (see training/DATA.md)."
            )
        raise ValueError("data.data_dir and data.embeddings_dir are required.")

    dd = Path(data_cfg["data_dir"]).expanduser().resolve()
    ed = Path(data_cfg["embeddings_dir"]).expanduser().resolve()
    if not dd.is_dir():
        raise FileNotFoundError(f"data_dir not found: {dd}")
    if not ed.is_dir():
        raise FileNotFoundError(f"embeddings_dir not found: {ed}")

    pairs: List[Tuple[Path, Path]] = []
    for p in sorted(dd.iterdir()):
        if p.suffix.lower() != ".jpg":
            continue
        npy = ed / f"{p.stem}.npy"
        if npy.is_file():
            pairs.append((p, npy))

    if not pairs:
        raise RuntimeError(
            "No training samples found. Check data_dir, embeddings_dir, and that each .jpg stem "
            "has a matching .npy in embeddings_dir."
        )
    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    rng = random.Random(seed)
    shuffled = list(pairs)
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


def save_split_manifest_pairs(
    out_path: Path,
    train: List[Tuple[Path, Path]],
    val: List[Tuple[Path, Path]],
    test: List[Tuple[Path, Path]],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def ser(ps: List[Tuple[Path, Path]]) -> List[Dict[str, str]]:
        return [{"jpg": str(a), "npy": str(b)} for a, b in ps]

    payload = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "train": ser(train),
        "val": ser(val),
        "test": ser(test),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def load_split_manifest_pairs(path: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    data = json.loads(path.read_text())

    def des(entries: List[Any]) -> List[Tuple[Path, Path]]:
        out: List[Tuple[Path, Path]] = []
        for e in entries:
            if isinstance(e, str):
                p = Path(e)
                out.append((p, p.with_suffix(".npy")))
            else:
                out.append((Path(e["jpg"]), Path(e["npy"])))
        return out

    return des(data["train"]), des(data["val"]), des(data["test"])


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

    def apply_coords(self, coords: np.ndarray, orig_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> np.ndarray:
        oh, ow = orig_hw
        nh, nw = new_hw
        scale = nw / ow
        out = coords.astype(np.float32) * scale
        return out


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
    """Paired (.jpg, teacher .npy) for encoder distillation."""

    def __init__(self, pairs: Sequence[Tuple[Path | str, Path | str]], img_size: int = 1024) -> None:
        self.pairs = [(Path(a), Path(b)) for a, b in pairs]
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        jpg, npy_path = self.pairs[index]
        img = cv2.imread(str(jpg))
        if img is None:
            raise FileNotFoundError(str(jpg))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = preprocess_image(img, self.img_size)
        feat = np.load(str(npy_path))
        feat = np.squeeze(feat)
        if feat.ndim != 3:
            raise ValueError(f"Expected teacher .npy (C,H,W) after squeeze, got {feat.shape} for {npy_path}")
        t = torch.from_numpy(feat.astype(np.float32))
        return x, t, str(jpg)


def subset_encoder_from_pairs(full: SA1BEncoderDataset, pairs: Sequence[Tuple[Path, Path]]) -> Subset:
    want = {(a.resolve(), b.resolve()) for a, b in pairs}
    indices = [i for i, pr in enumerate(full.pairs) if (pr[0].resolve(), pr[1].resolve()) in want]
    if len(indices) != len(pairs):
        raise RuntimeError(f"subset_encoder_from_pairs: expected {len(pairs)} matches, got {len(indices)}")
    return Subset(full, indices)


def combined_mask_from_json(data: dict, out_h: int, out_w: int) -> np.ndarray:
    if mask_util is None:
        raise ImportError("pycocotools is required for mask JSON; pip install pycocotools")
    m = np.zeros((out_h, out_w), dtype=np.float32)
    anns = data.get("annotations")
    if not anns and "segmentation" in data:
        anns = [data]
    if not anns:
        raise ValueError("No annotations/segmentation in JSON")
    for ann in anns:
        seg = ann.get("segmentation") if isinstance(ann, dict) else None
        if isinstance(seg, dict) and "counts" in seg:
            dec = mask_util.decode(seg)
            if dec.ndim == 3:
                dec = dec.transpose(2, 0, 1).max(axis=0)
            if dec.shape[0] != out_h or dec.shape[1] != out_w:
                dec = cv2.resize(dec.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            m = np.maximum(m, (dec > 0).astype(np.float32))
        elif isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, float, int)):
            rles = mask_util.frPyObjects(seg, out_h, out_w)
            for rle in rles:
                dec = mask_util.decode(rle)
                m = np.maximum(m, dec.astype(np.float32))
    if float(m.max()) <= 0:
        raise ValueError("Decoded empty mask from JSON")
    return (m > 0.5).astype(np.float32)


def resolve_annotation_json(jpg: Path, data_cfg: dict) -> Path:
    root = Path(data_cfg["annotation_root"]).expanduser().resolve()
    cand = root / f"{jpg.stem}.json"
    if cand.is_file():
        return cand
    cand = root / jpg.parent.name / f"{jpg.stem}.json"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"No annotation JSON for {jpg} under {root}")


class SA1BSamDataset(Dataset):
    """
    Full-SAM training sample: SAM-style image tensor (0–255 float CHW before preprocess in forward),
    box prompt from mask bbox, low-res (256) target mask.
    """

    def __init__(
        self,
        jpg_paths: Sequence[Path | str],
        data_cfg: dict,
        img_size: int = 1024,
        low_res: int = 256,
    ) -> None:
        self.paths = [Path(p) for p in jpg_paths]
        self.data_cfg = data_cfg
        self.img_size = img_size
        self.low_res = low_res

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        jpg = self.paths[index]
        img_bgr = cv2.imread(str(jpg))
        if img_bgr is None:
            raise FileNotFoundError(str(jpg))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        rs = ResizeLongestSide(self.img_size)
        rgb_s = rs.apply_image(rgb)
        nh, nw = rgb_s.shape[:2]
        scale_h, scale_w = nh / oh, nw / ow

        jpath = resolve_annotation_json(jpg, self.data_cfg)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        mask_full = combined_mask_from_json(data, oh, ow)
        mask_s = cv2.resize(mask_full, (nw, nh), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(mask_s > 0.5)
        if len(xs) == 0:
            raise ValueError(f"Empty mask after resize for {jpg}")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)

        sam_image = torch.from_numpy(rgb_s).permute(2, 0, 1).float()
        low_tgt = cv2.resize(mask_s, (self.low_res, self.low_res), interpolation=cv2.INTER_NEAREST)
        low_tgt_t = torch.from_numpy(low_tgt).float().unsqueeze(0)

        return {
            "image": sam_image,
            "original_size": (oh, ow),
            "boxes": box.unsqueeze(0),
            "low_res_mask_gt": low_tgt_t,
            "path": str(jpg),
        }


def build_datasets(
    data_cfg: dict,
    img_size: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_manifest: Path | None,
    split_manifest_out: Path | None,
):
    all_pairs = collect_encoder_pairs(data_cfg)
    full = SA1BEncoderDataset(all_pairs, img_size=img_size)

    if split_manifest is not None and split_manifest.is_file():
        train_p, val_p, test_p = load_split_manifest_pairs(split_manifest)
        meta = {"source": "manifest", "path": str(split_manifest)}
    else:
        train_p, val_p, test_p = split_pairs(all_pairs, seed, train_frac, val_frac, test_frac)
        meta = {
            "source": "fresh_split",
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
        }
        if split_manifest_out is not None:
            save_split_manifest_pairs(
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

    train_ds = subset_encoder_from_pairs(full, train_p)
    val_ds = subset_encoder_from_pairs(full, val_p)
    test_ds = subset_encoder_from_pairs(full, test_p)
    meta["counts"] = {"train": len(train_p), "val": len(val_p), "test": len(test_p)}

    jpg_train = [p[0] for p in train_p]
    jpg_val = [p[0] for p in val_p]
    jpg_test = [p[0] for p in test_p]

    return (
        full,
        train_ds,
        val_ds,
        test_ds,
        meta,
        {"train": jpg_train, "val": jpg_val, "test": jpg_test},
    )


def build_sam_loaders(
    data_cfg: dict,
    jpg_splits: Dict[str, List[Path]],
    batch_size: int,
    num_workers: int,
    distributed_sampler_train: Any,
    device_is_cuda: bool,
):
    if not data_cfg.get("annotation_root"):
        return None, None, None
    train_ds = SA1BSamDataset(jpg_splits["train"], data_cfg)
    val_ds = SA1BSamDataset(jpg_splits["val"], data_cfg)
    test_ds = SA1BSamDataset(jpg_splits["test"], data_cfg)

    def collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(distributed_sampler_train is None),
        sampler=distributed_sampler_train,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device_is_cuda,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device_is_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device_is_cuda,
    )
    return train_loader, val_loader, test_loader
