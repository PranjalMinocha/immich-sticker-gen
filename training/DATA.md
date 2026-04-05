# Data layout and object storage

## Recommended flow (Chameleon): sync to local disk, then Docker

On the GPU host, **`training/setup_host.sh`** (see README):

1. Configures **rclone** for your S3-compatible bucket (e.g. `objstore-proj28`).
2. **`rclone sync`** `Raw-Data/` and `Teacher-Embeddings/` into **`LOCAL_DATA_ROOT`** (default `~/training-data` — point this at instance **NVMe / scratch** if you have a larger or faster volume).
3. If **`Raw-Data/sa-1b-sample.tar.gz`** exists, extracts it once into **`Raw-Data/extracted/`** (marker file avoids repeat work; delete the marker and re-run to re-extract after a new tarball).
4. Optionally sets **`RCLONE_ENABLE_MOUNT=1`** to **FUSE-mount** the whole bucket for browsing; training I/O should use the **synced paths**, not the mount.

The training container bind-mounts that directory read-only, e.g. **`-v ~/training-data:/data:ro`**. In YAML set **`data.data_dir`** and **`data.embeddings_dir`** to the matching paths under **`/data`** (see **`configs/chameleon_docker.yaml`**).

### Bare metal on the instance (no Docker)

**`/data/...` does not exist** unless you create it (e.g. `sudo mkdir -p /data && sudo ln -s "$HOME/training-data" /data`). Otherwise copy a config that uses **host paths**, for example:

```yaml
data:
  data_dir: /home/cc/training-data/Raw-Data/extracted
  embeddings_dir: /home/cc/training-data/Teacher-Embeddings/sa_000000
  annotation_root: /home/cc/training-data/Raw-Data/extracted   # or wherever your JSON lives
```

Adjust **`cc`** and shard folder names to match **`setup_host.sh`** / **`LOCAL_DATA_ROOT`**. Use the **same paths** for `python3 prebuild_*.py --config ...` and for `train.py`.

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

## Precomputed SAM instance index (optional, faster startup)

Building the per-instance list scans every JPG and decodes every mask once; that can take a long time. You can **precompute once** and reuse:

- **Default file** (auto-loaded if present): **`{data_dir}/sam_instance_index/sam_instances_v1.json`**
- **Or** set **`data.sam_instance_index`** in YAML to an explicit path.

The index must match **`data_dir`**, **`embeddings_dir`**, **`annotation_root`**, and the same **split** definition as training (**`data.split_manifest`** path, or **`seed` + `train_frac` / `val_frac` / `test_frac`**). If you change any of those or the masks, rebuild the file.

**One-off script** (not in git — see repo **`.gitignore`**): copy **`training/prebuild_sam_instance_index.py`** onto the host (or create it from your checkout before it was ignored), then:

```bash
cd ~/immich-sticker-gen/training
python3 prebuild_sam_instance_index.py --config ~/training_out/run.yaml
# optional: --out /data/Raw-Data/extracted/sam_instance_index/sam_instances_v1.json
```

Use the **same YAML** as `train.py` so paths and split rules match. After the JSON exists under **`data_dir`**, Docker training starts without rescanning all images.

## Subsampling instance rows (shorter epochs, no re-index)

Optional **`data.sam_instance_frac`** in **`(0, 1]`** (default **`1.0`**): after loading the full train/val lists (from the index or a live scan), keep a **random subset** of that fraction for **both** train and val. Reproducible for a given **`data.seed`**. Does **not** change the JSON on disk.

**Test split:** the **test** loader always uses at most **`1000`** instance rows for final IoU, chosen with a **fixed RNG seed** in code (independent of **`data.seed`**) so eval is comparable across runs. If the full test split has fewer than 1000 rows, all are used.

MLflow logs **`sam_instance_frac`**, **`sam_train_instances_effective`**, **`sam_val_instances_effective`**, **`sam_test_instances_effective`**.
