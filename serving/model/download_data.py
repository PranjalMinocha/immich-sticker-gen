#!/usr/bin/env python3
"""
Download a 50-image SA-1B subset and the MobileSAM checkpoint.
Run once; subsequent runs skip already-downloaded files.
"""
import os, json, tarfile, requests
from pathlib import Path
from PIL import Image

SHARD_URL = (
    "https://scontent.xx.fbcdn.net/m1/v/t6/"
    "An_YmP5OIPXun-vu3hkckAZZ2s4lPYoVkiyvCcWiVY21mu1Ng5_1HeCa2CWiSTsskj8H"
    "Q8bN013HxNpYDdSC_7jWQq_svcg.tar?_nc_gid&ccb=10-5&oh=00_Af08ZycNdYdNMlM"
    "CW5sVXhQ0W2iYlA4GO1vtsjm6IY-yYw&oe=69F57028&_nc_sid=0fdd51"
)
CHECKPOINT_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt"
)

DATA_DIR   = Path(os.environ.get("DATA_DIR", "/data"))
MAX_IMAGES = 50


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"  → saved {dest}")


def prepare_sa1b_subset(max_images: int = MAX_IMAGES):
    images_dir = DATA_DIR / "images"
    anns_dir   = DATA_DIR / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    anns_dir.mkdir(parents=True, exist_ok=True)

    tar_path      = DATA_DIR / "sa1b_shard.tar"
    manifest_path = DATA_DIR / "manifest.json"

    # skip if already extracted
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if len(manifest) >= max_images:
            print(f"Subset already ready ({len(manifest)} pairs). Skipping.")
            return manifest[:max_images]

    if not tar_path.exists():
        download_file(SHARD_URL, tar_path)
    else:
        print("Shard already downloaded. Skipping.")

    print("Extracting image/json pairs ...")
    saved, pending_json = [], {}
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for m in members:
            if m.isfile() and m.name.endswith(".json"):
                f = tar.extractfile(m)
                if f:
                    pending_json[Path(m.name).stem] = f.read()
        for m in members:
            if not (m.isfile() and m.name.endswith(".jpg")):
                continue
            stem = Path(m.name).stem
            if stem not in pending_json:
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            img = Image.open(f).convert("RGB")
            img_path = images_dir / f"{stem}.jpg"
            ann_path = anns_dir   / f"{stem}.json"
            img.save(img_path, quality=95)
            ann_path.write_bytes(pending_json[stem])
            saved.append({
                "id": stem,
                "image_path": str(img_path),
                "annotation_path": str(ann_path),
            })
            if len(saved) >= max_images:
                break

    manifest_path.write_text(json.dumps(saved, indent=2))
    print(f"Saved {len(saved)} pairs → {DATA_DIR}")
    return saved


def download_checkpoint():
    ckpt_path = DATA_DIR / "mobile_sam.pt"
    if ckpt_path.exists():
        print("Checkpoint already exists. Skipping.")
    else:
        download_file(CHECKPOINT_URL, ckpt_path)
    return str(ckpt_path)


if __name__ == "__main__":
    prepare_sa1b_subset()
    download_checkpoint()
    print("Done.")