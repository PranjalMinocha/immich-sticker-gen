# Data layout and object storage

## Recommended flow (Chameleon): sync to local disk, then Docker

On the GPU host, **`training/setup_host.sh`** (see README):

1. Configures **rclone** for your S3-compatible bucket (e.g. `objstore-proj28`).
2. **`rclone sync`** `Raw-Data/` and `Teacher-Embeddings/` into **`LOCAL_DATA_ROOT`** (default `~/training-data` — point this at instance **NVMe / scratch** if you have a larger or faster volume).
3. If **`Raw-Data/sa-1b-sample.tar.gz`** exists, extracts it once into **`Raw-Data/extracted/`** (marker file avoids repeat work; delete the marker and re-run to re-extract after a new tarball).
4. Optionally sets **`RCLONE_ENABLE_MOUNT=1`** to **FUSE-mount** the whole bucket for browsing; training I/O should use the **synced paths**, not the mount.

The training container bind-mounts that directory read-only, e.g. **`-v ~/training-data:/data:ro`**. In YAML set **`data.data_dir`** and **`data.embeddings_dir`** to the matching paths under **`/data`** (see **`configs/chameleon_docker.yaml`**).

## How training reads data

The training code **does not** load your entire dataset into RAM. It uses PyTorch `DataLoader` workers that **open each `.jpg` / `.npy` (and optional `.json`) on demand** from the configured paths.

- **Memory**: bounded roughly by batch size × image/embedding tensors, not by total corpus size.
- **Throughput**: **local SSD** after `rclone sync` avoids per-file latency from object storage; FUSE mounts are optional and can be slower for random access.

**Tar archives (`*.tar.gz`) are not read directly.** `setup_host.sh` extracts `sa-1b-sample.tar.gz` into `Raw-Data/extracted/` (or extract manually / adjust `SA1B_SAMPLE_TAR` / `RAW_EXTRACT_SUBDIR` in env).

## YAML: `data_dir` and `embeddings_dir`

| Key | Contents |
|-----|----------|
| **`data_dir`** | Folder scanned for **`*.jpg`**. Optional mask JSON for SAM modes can live here too (**`{stem}.json`** next to **`{stem}.jpg`**) or under **`annotation_root`**. |
| **`embeddings_dir`** | Folder containing teacher **`{stem}.npy`** for each training **`{stem}.jpg`** (same basename). Can be any path (e.g. a folder under `Teacher-Embeddings/`). |

Example (Docker mount `~/training-data` → `/data`):

```yaml
data:
  data_dir: /data/Raw-Data/extracted
  embeddings_dir: /data/Teacher-Embeddings/sa_000000
```

If the tarball unpacks with an **extra directory level**, set **`data_dir`** (and **`annotation_root`** for full-SAM) to the directory that **actually contains the `.jpg` files**.

## Masks for full-SAM / test IoU

Set **`data.annotation_root`** and place COCO-style JSON per image (RLE in `annotations[].segmentation` or polygon lists). **`full_sam` training uses one row per annotation**: each step supervises a single instance mask with a box prompt from that annotation’s COCO **`bbox`** (or, if missing, a tight box on the decoded mask). Image-level train/val/test splits are unchanged; the dataset length is the total number of valid instances across images in each split.

Lookup order (see `dataset_sa1b.resolve_annotation_json`):

1. **`annotation_root/{stem}.json`**
2. **`annotation_root/{jpg_parent_name}/{stem}.json`**
