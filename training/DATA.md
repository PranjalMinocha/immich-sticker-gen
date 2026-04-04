# Data layout and object storage

## Recommended flow (Chameleon): sync to local disk, then Docker

On the GPU host, **`training/setup_host.sh`** (see README):

1. Configures **rclone** for your S3-compatible bucket (e.g. `objstore-proj28`).
2. **`rclone sync`** `Raw-Data/` and `Teacher-Embeddings/` into **`LOCAL_DATA_ROOT`** (default `~/training-data` — point this at instance **NVMe / scratch** if you have a larger or faster volume).
3. If **`Raw-Data/sa-1b-sample.tar.gz`** exists, extracts it once into **`Raw-Data/extracted/`** (marker file avoids repeat work; delete the marker and re-run to re-extract after a new tarball).
4. Optionally sets **`RCLONE_ENABLE_MOUNT=1`** to **FUSE-mount** the whole bucket for browsing; training I/O should use the **synced paths**, not the mount.

The training container bind-mounts that directory read-only, e.g. **`-v ~/training-data:/data:ro`**, and YAML uses **`split_teacher`** with `image_root` / `teacher_root` under `/data/...` (see **`configs/chameleon_docker_split_teacher.yaml`**).

## How training reads data

The training code **does not** load your entire dataset into RAM. It uses PyTorch `DataLoader` workers that **open each `.jpg` / `.npy` (and optional `.json`) on demand** from the configured paths.

That means:

- **Memory**: bounded roughly by batch size × image/embedding tensors, not by total corpus size.
- **Throughput**: **local SSD** after `rclone sync` avoids per-file latency from object storage; FUSE mounts are optional and can be slower for random access.

**Tar archives (`*.tar.gz`) are not read directly.** `setup_host.sh` extracts `sa-1b-sample.tar.gz` into `Raw-Data/extracted/` (or extract manually / adjust `SA1B_SAMPLE_TAR` / `RAW_EXTRACT_SUBDIR` in env).

If the archive unpacks with an **extra top-level folder** (e.g. `extracted/my_prefix/sa_000000/...`), set **`data.image_root`** to include that prefix so `.../<shard>/*.jpg` resolves.

## Layouts

### `colocated` (default)

```
data.root / <shard> / foo.jpg
data.root / <shard> / foo.npy
```

### `split_teacher` (matches Raw-Data + Teacher-Embeddings buckets)

After host prep + extract:

```yaml
data:
  layout: split_teacher
  image_root: /data/Raw-Data/extracted
  teacher_root: /data/Teacher-Embeddings
  shard_dirs: [sa_000000]
```

Requires `image_root/<shard>/<id>.jpg` and `teacher_root/<shard>/<id>.npy` with the **same stem**.

### Masks for full-SAM modes / test IoU

Set `data.annotation_root` and place COCO-style JSON per image (RLE in `annotations[].segmentation` or polygon lists). Resolution:

- `annotation_root/<shard>/<stem>.json`, or
- `annotation_root/<stem>.json`

See `dataset_sa1b.resolve_annotation_json` for lookup order.
