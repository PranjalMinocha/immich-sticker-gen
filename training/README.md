# Training — unified MobileSAM workflows

Training is controlled by **two YAML switches** (see below): **`training.mode`** and **`training.use_pretrained`** (+ path). The default path **distills a TinyViT encoder** (student always **random init** in code today) vs teacher **`.npy`**, then **assembles** a full MobileSAM checkpoint for MLflow. **`training.mode: full_sam`** fine-tunes the **entire** SAM on masks. **Distill then segment** = run `encoder_distill`, then `full_sam` with **`training.pretrained_checkpoint_path`** set to the first run’s **`mobile_sam_full.pt`**. All runs are **config-only** and log to **[MLflow](https://mlflow.org/)** via **[`train.py`](train.py)**.

**Data & object storage:** see **[`DATA.md`](DATA.md)** (`rclone sync` to local disk, optional FUSE mount, **`data_dir`** + **`embeddings_dir`**, tarball extract). **End-to-end pipeline diagram:** **[`PIPELINE.md`](PIPELINE.md)**.

## Layout

| File / directory | Role |
|------------------|------|
| [`train.py`](train.py) | **Main entry:** `training.mode` + `training.use_pretrained` / `pretrained_checkpoint_path` |
| [`training_core.py`](training_core.py) | Distributed setup, flatten_cfg, TinyViT import path, encoder loss, encoder eval |
| [`sam_utils.py`](sam_utils.py) | Trainable SAM forward, merge encoder into checkpoint, seg loss / IoU |
| [`dataset_sa1b.py`](dataset_sa1b.py) | Splits, `data_dir` + `embeddings_dir` pairs, optional mask JSON for SAM modes |
| [`DATA.md`](DATA.md) | Object storage sync, layouts, tar extract |
| [`PIPELINE.md`](PIPELINE.md) | Mermaid flowchart + pipeline summary |
| [`configs/tinyvit_baseline.yaml`](configs/tinyvit_baseline.yaml) | Encoder distillation template (`training.pretrained_checkpoint_path` when `use_pretrained: true`) |
| [`configs/tinyvit_local_sa1b.yaml`](configs/tinyvit_local_sa1b.yaml) | Local smoke-test paths |
| [`configs/example_full_sam.yaml`](configs/example_full_sam.yaml) | Full-model mask supervision (after distill: `pretrained_checkpoint_path` → prior `mobile_sam_full.pt`) |
| [`requirements.txt`](requirements.txt) | Python deps (install **PyTorch** separately for your CUDA / ROCm stack) |
| [`Dockerfile`](Dockerfile) | ROCm 6.0 training image — build from **repository root** |
| [`setup_host.sh`](setup_host.sh) | Chameleon host: rclone, **sync** `Raw-Data` + `Teacher-Embeddings` to local disk, extract sample tarball, optional mount |
| [`configs/chameleon_docker.yaml`](configs/chameleon_docker.yaml) | Docker paths when data is mounted at `/data` |

**TinyViT import path:** set `mobilesam_root` in YAML, or env `MOBILESAM_ROOT` / `IMMICH_MS_ROOT`, or clone [MobileSAM-pytorch](https://github.com/ChaoningZhang/MobileSAM-pytorch) so `../MobileSAM-pytorch/MobileSAM` exists next to this repo.

### Config: mode + pretrained weights

| Key | Values | Role |
|-----|--------|------|
| **`training.mode`** | `encoder_distill` \| `full_sam` | **Encoder path:** distill TinyViT vs teacher `.npy`, merge into a SAM scaffold. **Full path:** train all SAM components on mask loss. |
| **`training.use_pretrained`** | `true` (default) \| `false` | **`true`:** load **`training.pretrained_checkpoint_path`** (official MobileSAM, your `mobile_sam_full.pt`, etc.). **`false`:** random-init MobileSAM scaffold (no `.pt`). |
| **`training.pretrained_checkpoint_path`** | filesystem path | Required when **`use_pretrained: true`**. Omit or ignore when **`false`**. |

**Student TinyViT** in **`encoder_distill`** is always **randomly initialized**; **`use_pretrained`** only controls the **SAM `.pt`** used for merge / full-model init.

| `training.mode` | What it does | MLflow artifact |
|-----------------|--------------|-----------------|
| `encoder_distill` | TinyViT (random) vs teacher `.npy`; merge into SAM from checkpoint or scratch scaffold | `checkpoints/mobile_sam_full.pt` (+ split manifest) |
| `full_sam` | BCE+Dice on low-res masks; `data.annotation_root` + JSON | `checkpoints/mobile_sam_full.pt` |

**System metrics:** each epoch logs CPU/RAM/disk (via **psutil**), GPU memory, and optional **`gpu_util_percent_rocm_smi`** when `rocm-smi` is available.

**`full_sam` validation previews:** after each epoch’s val IoU, rank 0 logs **`train.val_preview_samples`** (default **3**) PNGs to MLflow under **`val_previews/epoch_XXXX/`**: RGB image, **box prompt** (same bbox-from-GT as training), predicted mask (green overlay), GT mask (red contour). Set **`val_preview_samples: 0`** to turn off.

**`encoder_distill` merged-SAM previews:** with **`data.annotation_root`** and **`train.val_preview_samples` > 0**, after val metrics rank 0 merges TinyViT into the SAM scaffold, logs **`val_previews_merged_sam/epoch_XXXX/`**.

---

## 1. Set up a training instance (Chameleon)

These steps assume a **GPU bare-metal** node (e.g. ROCm image at CHI@TACC) and that **large data** lives in **team object storage**, not only on local disk.

### 1.1 Bring up the node

- Reserve a lease and start a server (e.g. via course Jupyter + Chameleon bindings, or the CLI).
- Attach a **floating IP** and SSH in as your user (often `cc`).

### 1.2 Install Docker (for graded container runs)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# use `sudo docker` until you log out/in, or: newgrp docker
```

### 1.3 Secrets and staging data on local disk

On the **server**, create **`~/training.env`** (mode `600`) with at least:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Optional overrides (defaults match `setup_host.sh`):

| Variable | Purpose |
|----------|---------|
| `S3_ENDPOINT_URL`, `OBJSTORE_BUCKET`, `RCLONE_REMOTE_NAME` | Object store connection |
| `LOCAL_DATA_ROOT` | Where **`Raw-Data`** and **`Teacher-Embeddings`** are synced (default `~/training-data`; use fast instance storage if available) |
| `OBJSTORE_RAW_PREFIX`, `OBJSTORE_TEACHER_PREFIX` | Bucket folder names (default `Raw-Data`, `Teacher-Embeddings`) |
| `SA1B_SAMPLE_TAR`, `RAW_EXTRACT_SUBDIR` | Tarball name and extract target under `Raw-Data/` |
| `RCLONE_ENABLE_MOUNT` | Set to `1` to also **FUSE-mount** the whole bucket (off by default; training should use **`LOCAL_DATA_ROOT`**) |
| `RCLONE_MOUNTPOINT` | Mount path when `RCLONE_ENABLE_MOUNT=1` |

From the cloned repo, run host prep (installs rclone, **syncs** bucket prefixes, **extracts** `sa-1b-sample.tar.gz` once into `Raw-Data/extracted/`):

```bash
cd ~/immich-sticker-gen
git pull
bash training/setup_host.sh
```

Verify local layout:

```bash
ls "${LOCAL_DATA_ROOT:-$HOME/training-data}/Raw-Data/extracted" | head
ls "${LOCAL_DATA_ROOT:-$HOME/training-data}/Teacher-Embeddings" | head
```

For Docker, bind-mount that directory at **`/data`** and set **`data.data_dir`** and **`data.embeddings_dir`** (see **`configs/chameleon_docker.yaml`** and **`DATA.md`**).

### 1.4 Clone repo and MobileSAM on the instance

```bash
cd ~
# Replace with your team’s GitHub URL for this repository.
git clone https://github.com/PranjalMinocha/immich-sticker-gen.git
git clone --recursive https://github.com/liuguoyou/MobileSAM-pytorch.git
```

Set **`MOBILESAM_ROOT`** to `~/MobileSAM-pytorch/MobileSAM` (or pass `mobilesam_root` in YAML).

### 1.5 MLflow

Point every run at the **team tracking server** (replace with your URI):

```bash
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000
```

---

## 2. Change configuration

All hyperparameters and paths should come from **one YAML file** per run (no one-off script copies for each experiment).

1. **Copy a template:**

   ```bash
   cp training/configs/tinyvit_baseline.yaml ~/training_out/run.yaml
   ```

2. **Edit** `~/training_out/run.yaml` (or use `sed` / your editor). Important keys:

   | Key area | Examples |
   |----------|----------|
   | `training.mode`, `training.use_pretrained`, `training.pretrained_checkpoint_path` | `encoder_distill` / `full_sam`; load `.pt` or train scaffold from scratch |
   | `data.data_dir` + `data.embeddings_dir` (see `DATA.md`) |
   | `output.dir` | Where checkpoints / logs go; use `/out` in Docker when mounting `~/training_out:/out` |
   | `mobilesam_root` | Path to `MobileSAM` package (or rely on `MOBILESAM_ROOT`) |
   | `train.epochs`, `train.batch_size`, `train.lr`, … | Experiment knobs |
   | `train.distributed_backend` | On some AMD setups use `gloo` if NCCL fails |

3. **Smoke test:** set `train.epochs: 1` to verify paths and MLflow before long runs.

---

## 3. Start a training run

**Working directory for bare-metal:** `training/` (so imports and relative config paths resolve).

### 3.1 Bare metal (single GPU)

```bash
cd ~/immich-sticker-gen
git pull
pip install -r training/requirements.txt
# PyTorch ROCm example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

export MOBILESAM_ROOT=~/MobileSAM-pytorch/MobileSAM
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000

cd training
python3 train.py --config /path/to/run.yaml
```

### 3.2 Bare metal (2× GPU, e.g. MI100 ×2)

```bash
cd ~/immich-sticker-gen/training
export MOBILESAM_ROOT=~/MobileSAM-pytorch/MobileSAM
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000

torchrun --nproc_per_node=2 train.py --config /path/to/run.yaml
```

### 3.3 Docker (ROCm) — matches graded “train in container” flow

Build from **repository root** with **BuildKit** so **rebuilds** reuse cached layers (especially PyTorch) when only parts of the repo change:

```bash
cd ~/immich-sticker-gen
DOCKER_BUILDKIT=1 docker build -f training/Dockerfile -t immich-sticker-train:rocm .
```

If you **only edit the mounted config** (e.g. `~/training_out/run.yaml` → `/out/run.yaml` inside the container), **start a new container**; you do **not** need to rebuild the image. Rebuild when **`training/`** code or **`training/requirements.txt`** changes.

Example run: mount **staged local data** (same tree `setup_host.sh` created under `LOCAL_DATA_ROOT`, default `~/training-data`):

```bash
DATA_ROOT="${LOCAL_DATA_ROOT:-$HOME/training-data}"
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$DATA_ROOT:/data:ro" \
  -v ~/MobileSAM-pytorch/MobileSAM:/mobilesam:ro \
  -v ~/training_out:/out \
  -e MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000 \
  -e MOBILESAM_ROOT=/mobilesam \
  immich-sticker-train:rocm \
  torchrun --nproc_per_node=2 train.py --config /out/run.yaml
```

Copy and edit **`training/configs/chameleon_docker.yaml`** into `/out/run.yaml` (set **`training.pretrained_checkpoint_path`**, MLflow URI), or adjust **`data.data_dir`** / **`data.embeddings_dir`** to match `/data`.

---

## 4. Local development (laptop / workstation)

Not counted for the Chameleon graded runs, but useful for debugging.

```bash
cd training
pip install -r requirements.txt   # plus torch for your platform (CPU/CUDA/ROCm)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # or team server
python train.py --config configs/tinyvit_local_sa1b.yaml
```

Adjust `tinyvit_local_sa1b.yaml` for your paths.

---

## 5. How long a run takes

Depends on **image count**, **epochs**, **batch size**, and **GPU count**. Use tqdm’s **epoch 1** wall time × `epochs` as a rough estimate.

| Scenario | Ballpark |
|----------|----------|
| **1-epoch smoke test** | Often minutes |
| **Full `sa_000000`-scale, ~8 epochs** | Often tens of minutes to a few hours |

---

## 6. Course alignment (training role)

- Runs that count for the project should execute **on Chameleon**, **inside a container**, with **MLflow** tracking.
- Keep **one training script** per framework; select candidates via **config**, not duplicate scripts.
- Log **params**, **quality metrics**, **cost metrics** (time/epoch, total time), and **environment** (GPU/torch) each run.

For project-wide context and Immich integration, see the [root README](../README.md).
