"""
SA-1B-style data: encoder distillation (.jpg + teacher .npy) and mask JSON for full-SAM training.

Full-SAM supervision is **per instance**: each sample is one JPEG, one annotation index, one box prompt
(from COCO bbox or tight box on the instance mask), and one binary mask for that annotation only.

Data paths: **data.data_dir** (JPEGs) and **data.embeddings_dir** (teacher **{stem}.npy**). See DATA.md.

Training reads files via the filesystem (e.g. rclone mount). It does not load the full dataset into RAM;
throughput depends on mount latency. Extract tar archives to a directory first — see training/DATA.md.
"""
from __future__ import annotations

import json
import csv
import random
import sys
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

SAM_INSTANCE_INDEX_VERSION = 1
DEFAULT_SAM_INSTANCE_INDEX_REL = Path("sam_instance_index") / "sam_instances_v1.json"

# MobileSAM / SAM: mask_decoder 256² aligns to the padded square fed to the image encoder (1024²).
SAM_ENCODER_PAD_SIDE = 1024
SAM_MASK_LOW_RES = 256


def resolved_sam_instance_index_path(data_cfg: dict) -> Path | None:
    """
    Path to precomputed instance list if configured and file exists.
    - data.sam_instance_index: explicit file (required to exist if set)
    - else: data_dir/sam_instance_index/sam_instances_v1.json if present
    """
    raw = data_cfg.get("sam_instance_index")
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"data.sam_instance_index not found: {p}")
        return p
    dd = Path(data_cfg["data_dir"]).expanduser().resolve()
    cand = (dd / DEFAULT_SAM_INSTANCE_INDEX_REL).resolve()
    if cand.is_file():
        return cand
    return None


def _jpg_rel_to_data_dir(jpg: Path, data_dir: Path) -> str:
    try:
        return str(jpg.resolve().relative_to(data_dir.resolve()))
    except ValueError:
        return str(jpg.resolve())


def _jpg_from_rel(rel: str, data_dir: Path) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    return (data_dir / p).resolve()


def load_sam_instance_index_entries(
    index_path: Path, data_cfg: dict, split: str
) -> List[Tuple[Path, int]]:
    """
    Load (jpg_path, ann_idx) list for a split.

    All filesystem roots come from ``data_cfg`` (YAML). The index only supplies
    ``jpg_rel`` paths relative to ``data.data_dir`` and split membership; stored
    ``data_dir`` / ``embeddings_dir`` / ``annotation_root`` in the JSON are ignored
    so the same file works across host vs container mount paths.
    """
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != SAM_INSTANCE_INDEX_VERSION:
        raise ValueError(f"Unsupported sam_instance_index format in {index_path}")

    dd = Path(data_cfg["data_dir"]).expanduser().resolve()

    cfg_manifest = data_cfg.get("split_manifest")
    cfg_m_path = Path(cfg_manifest).resolve() if cfg_manifest else None
    idx_manifest_raw = payload.get("split_manifest")
    idx_m_path = Path(idx_manifest_raw).resolve() if idx_manifest_raw else None

    if idx_m_path is not None:
        if cfg_m_path is None:
            raise ValueError(
                "sam_instance_index was built with split_manifest; set data.split_manifest in YAML "
                "(path is resolved from config only; it need not match the path stored in the index)."
            )
        if cfg_m_path != idx_m_path:
            print(
                f"sam_instance_index: split_manifest path in index ({idx_m_path}) differs from "
                f"config ({cfg_m_path}); using config path. Ensure the manifest content matches "
                f"how the index was built.",
                file=sys.stderr,
                flush=True,
            )
    else:
        seed = int(data_cfg.get("seed", 42))
        tf = float(data_cfg.get("train_frac", 0.7))
        vf = float(data_cfg.get("val_frac", 0.1))
        xf = float(data_cfg.get("test_frac", 0.2))
        if cfg_m_path is not None:
            raise ValueError(
                "sam_instance_index was built without split_manifest but training has data.split_manifest set. "
                "Rebuild the index with the same manifest, or drop split_manifest for training."
            )
        p_seed = payload.get("seed")
        if p_seed is None:
            raise ValueError("sam_instance_index missing seed (expected when built without split_manifest). Rebuild.")
        if (
            int(p_seed) != seed
            or abs(float(payload.get("train_frac", -1.0)) - tf) > 1e-6
            or abs(float(payload.get("val_frac", -1.0)) - vf) > 1e-6
            or abs(float(payload.get("test_frac", -1.0)) - xf) > 1e-6
        ):
            raise ValueError(
                "sam_instance_index seed/train_frac/val_frac/test_frac do not match data config. Rebuild the index."
            )

    sp = payload.get("splits", {}).get(split)
    if not sp:
        raise ValueError(f"No splits[{split!r}] in {index_path}")
    out: List[Tuple[Path, int]] = []
    for row in sp:
        out.append((_jpg_from_rel(row["jpg_rel"], dd), int(row["ann_idx"])))
    return out


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


def _resolve_image_uri_to_local_path(image_uri: str, data_cfg: dict) -> Path:
    raw = image_uri.strip()
    if raw.startswith("s3://"):
        local_root_raw = data_cfg.get("objstore_local_root")
        if not local_root_raw:
            raise ValueError(
                "CSV manifest contains s3:// image_uri but data.objstore_local_root is not set."
            )
        parts = raw.split("/", 3)
        if len(parts) < 4:
            raise ValueError(f"Invalid s3 URI in manifest: {raw}")
        key = parts[3]
        return (Path(local_root_raw).expanduser().resolve() / key).resolve()
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (Path(data_cfg["data_dir"]).expanduser().resolve() / p).resolve()


def load_csv_manifest_pairs(
    train_manifest_csv: Path,
    val_manifest_csv: Path,
    data_cfg: dict,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    embeddings_dir = Path(data_cfg["embeddings_dir"]).expanduser().resolve()

    def parse_one(path: Path) -> List[Tuple[Path, Path]]:
        out: List[Tuple[Path, Path]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV manifest has no header: {path}")
            image_col = None
            for cand in ("image_uri", "image_path", "jpg", "image"):
                if cand in reader.fieldnames:
                    image_col = cand
                    break
            if image_col is None:
                raise ValueError(f"CSV manifest missing image column in {path}; expected one of image_uri/image_path/jpg/image")

            for row in reader:
                image_raw = (row.get(image_col) or "").strip()
                if not image_raw:
                    continue
                jpg = _resolve_image_uri_to_local_path(image_raw, data_cfg)
                npy = embeddings_dir / f"{jpg.stem}.npy"
                out.append((jpg, npy))

        if not out:
            raise RuntimeError(f"No rows parsed from CSV manifest: {path}")
        return out

    train = parse_one(train_manifest_csv)
    val = parse_one(val_manifest_csv)
    test = list(val)
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


def _annotations_list_from_json(data: dict) -> List[dict]:
    anns = data.get("annotations")
    if not anns and "segmentation" in data and isinstance(data.get("segmentation"), (dict, list)):
        return [data] if isinstance(data, dict) else []
    if not anns:
        return []
    return [a for a in anns if isinstance(a, dict)]


def mask_from_ann_segmentation(ann: dict, out_h: int, out_w: int) -> np.ndarray | None:
    """Decode a single annotation's segmentation to a binary float mask (out_h, out_w), or None if empty."""
    if mask_util is None:
        raise ImportError("pycocotools is required for mask JSON; pip install pycocotools")
    seg = ann.get("segmentation")
    if seg is None:
        return None
    m = np.zeros((out_h, out_w), dtype=np.float32)
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
    else:
        return None
    if float(m.max()) <= 0:
        return None
    return (m > 0.5).astype(np.float32)


def list_instance_samples(
    jpg_paths: Sequence[Path],
    data_cfg: dict,
    *,
    progress_label: str = "",
) -> List[Tuple[Path, int]]:
    """
    One training row per (image, annotation_index) with a non-empty decoded mask.
    Splits stay image-level; this expands each split's JPG list into instance indices.
    """
    out: List[Tuple[Path, int]] = []
    ann_root = Path(data_cfg["annotation_root"]).expanduser().resolve()
    paths = [Path(p) for p in jpg_paths]
    n_img = len(paths)
    label = f" [{progress_label}]" if progress_label else ""
    print(
        f"SAM dataset: building instance index{label} over {n_img} images (read JPG + JSON + decode masks; can take several minutes)...",
        file=sys.stderr,
        flush=True,
    )
    # Frequent logs: first 5 images individually, then every 25 (7830//20 was ~391 = long silence).
    log_every = 25

    def _log_progress(done: int) -> None:
        print(
            f"SAM dataset{label}: {done}/{n_img} images scanned → {len(out)} instance samples",
            file=sys.stderr,
            flush=True,
        )

    for img_i, jpg in enumerate(paths):
        jpg = Path(jpg)
        if img_i == 0:
            print(
                f"SAM dataset{label}: starting {jpg.name} (if this line hangs, check this file + its JSON / mask decode cost)",
                file=sys.stderr,
                flush=True,
            )
        try:
            jpath = resolve_annotation_json(jpg, data_cfg)
        except FileNotFoundError:
            pass
        else:
            img_bgr = cv2.imread(str(jpg))
            if img_bgr is not None:
                oh, ow = img_bgr.shape[:2]
                with open(jpath, encoding="utf-8") as f:
                    data = json.load(f)
                anns = _annotations_list_from_json(data)
                for i, ann in enumerate(anns):
                    m = mask_from_ann_segmentation(ann, oh, ow)
                    if m is not None:
                        out.append((jpg, i))

        done = img_i + 1
        if done <= 5 or done % log_every == 0 or done == n_img:
            _log_progress(done)
    if not out:
        raise RuntimeError(
            f"No instance samples under annotation_root={ann_root}. "
            "Check JSON layout (annotations[].segmentation) and that each split JPG has a matching JSON."
        )
    print(
        f"SAM dataset{label}: done — {len(out)} instance samples from {n_img} images",
        file=sys.stderr,
        flush=True,
    )
    return out


def resolve_annotation_json(jpg: Path, data_cfg: dict) -> Path:
    root = Path(data_cfg["annotation_root"]).expanduser().resolve()
    cand = root / f"{jpg.stem}.json"
    if cand.is_file():
        return cand
    cand = root / jpg.parent.name / f"{jpg.stem}.json"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"No annotation JSON for {jpg} under {root}")


def _box_xyxy_resized_from_ann_or_mask(
    ann: dict, oh: int, ow: int, nh: int, nw: int, mask_s: np.ndarray
) -> torch.Tensor:
    """COCO bbox [x,y,w,h] in original pixels → xyxy in resized (nh,nw); else tight box on mask_s."""
    bb = ann.get("bbox")
    sx, sy = nw / ow, nh / oh
    if bb is not None and len(bb) == 4:
        x, y, w, h = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        x0, y0 = x * sx, y * sy
        x1, y1 = (x + w) * sx, (y + h) * sy
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
    else:
        ys, xs = np.where(mask_s > 0.5)
        if len(xs) == 0:
            raise ValueError("Empty instance mask for box fallback")
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
    return box


class SA1BSamDataset(Dataset):
    """
    Full-SAM training: one sample per (image, annotation index). RGB image (float CHW 0–255),
    box prompt from annotation COCO bbox (or tight box on the instance mask), low-res (256) GT mask.
    """

    def __init__(
        self,
        jpg_paths: Sequence[Path | str],
        data_cfg: dict,
        img_size: int = 1024,
        low_res: int = 256,
        progress_label: str = "",
        split: str | None = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.img_size = img_size
        self.low_res = low_res
        label = f" [{progress_label}]" if progress_label else ""
        idx_path = resolved_sam_instance_index_path(data_cfg)
        if idx_path is not None:
            if not split:
                raise ValueError(
                    "Precomputed SAM instance index is in use; pass split='train'|'val'|'test' to SA1BSamDataset."
                )
            print(
                f"SAM dataset{label}: loading {split} split from {idx_path} …",
                file=sys.stderr,
                flush=True,
            )
            self.samples = load_sam_instance_index_entries(idx_path, data_cfg, split)
            print(
                f"SAM dataset{label}: loaded {len(self.samples)} instance samples (cached index)",
                file=sys.stderr,
                flush=True,
            )
        else:
            self.samples = list_instance_samples(
                [Path(p) for p in jpg_paths],
                data_cfg,
                progress_label=progress_label,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        jpg, ann_idx = self.samples[index]
        img_bgr = cv2.imread(str(jpg))
        if img_bgr is None:
            raise FileNotFoundError(str(jpg))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        rs = ResizeLongestSide(self.img_size)
        rgb_s = rs.apply_image(rgb)
        nh, nw = rgb_s.shape[:2]

        jpath = resolve_annotation_json(jpg, self.data_cfg)
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        anns = _annotations_list_from_json(data)
        if ann_idx >= len(anns):
            raise IndexError(f"ann_idx {ann_idx} out of range for {jpg}")
        ann = anns[ann_idx]
        mask_full = mask_from_ann_segmentation(ann, oh, ow)
        if mask_full is None:
            raise ValueError(f"Empty mask for {jpg} ann {ann_idx}")
        mask_s = cv2.resize(mask_full, (nw, nh), interpolation=cv2.INTER_NEAREST)

        box = _box_xyxy_resized_from_ann_or_mask(ann, oh, ow, nh, nw, mask_s)

        sam_image = torch.from_numpy(rgb_s).permute(2, 0, 1).float()
        padded = np.zeros((SAM_ENCODER_PAD_SIDE, SAM_ENCODER_PAD_SIDE), dtype=np.float32)
        padded[0:nh, 0:nw] = mask_s.astype(np.float32, copy=False)
        low_tgt = cv2.resize(
            padded, (self.low_res, self.low_res), interpolation=cv2.INTER_NEAREST
        )
        low_tgt_t = torch.from_numpy(low_tgt).float().unsqueeze(0)

        return {
            "image": sam_image,
            "original_size": (oh, ow),
            "boxes": box.unsqueeze(0),
            "low_res_mask_gt": low_tgt_t,
            "path": str(jpg),
            "ann_idx": ann_idx,
        }


def build_datasets(
    data_cfg: dict,
    img_size: int,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_manifest: Path | None,
    train_manifest_csv: Path | None,
    val_manifest_csv: Path | None,
    split_manifest_out: Path | None,
):
    if train_manifest_csv is not None or val_manifest_csv is not None:
        if train_manifest_csv is None or val_manifest_csv is None:
            raise ValueError("Both data.train_manifest_csv and data.val_manifest_csv must be set together.")
        if not train_manifest_csv.is_file():
            raise FileNotFoundError(f"train manifest csv not found: {train_manifest_csv}")
        if not val_manifest_csv.is_file():
            raise FileNotFoundError(f"val manifest csv not found: {val_manifest_csv}")
        train_p, val_p, test_p = load_csv_manifest_pairs(train_manifest_csv, val_manifest_csv, data_cfg)
        all_pairs = []
        seen = set()
        for jpg_path, npy_path in train_p + val_p + test_p:
            key = (str(jpg_path.resolve()), str(npy_path.resolve()))
            if key in seen:
                continue
            seen.add(key)
            all_pairs.append((jpg_path, npy_path))
        full = SA1BEncoderDataset(all_pairs, img_size=img_size)
        meta = {
            "source": "manifest_csv",
            "train_manifest_csv": str(train_manifest_csv),
            "val_manifest_csv": str(val_manifest_csv),
            "test_policy": "test_equals_val",
        }
    elif split_manifest is not None and split_manifest.is_file():
        all_pairs = collect_encoder_pairs(data_cfg)
        full = SA1BEncoderDataset(all_pairs, img_size=img_size)
        train_p, val_p, test_p = load_split_manifest_pairs(split_manifest)
        meta = {"source": "manifest", "path": str(split_manifest)}
    else:
        all_pairs = collect_encoder_pairs(data_cfg)
        full = SA1BEncoderDataset(all_pairs, img_size=img_size)
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
    device_is_cuda: bool,
):
    if not data_cfg.get("annotation_root"):
        return None, None, None
    img_sz = int(data_cfg.get("image_size", 1024))
    train_ds = SA1BSamDataset(
        jpg_splits["train"], data_cfg, img_size=img_sz, progress_label="train", split="train"
    )
    val_ds = SA1BSamDataset(
        jpg_splits["val"], data_cfg, img_size=img_sz, progress_label="val", split="val"
    )
    test_ds = SA1BSamDataset(
        jpg_splits["test"], data_cfg, img_size=img_sz, progress_label="test", split="test"
    )

    def collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
