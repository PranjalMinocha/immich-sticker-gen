# Training — TinyViT encoder distillation

Stage-1 training **distills a TinyViT encoder** to match **ViT-H teacher embeddings** (MobileSAM-style): for each training image, a **`.npy`** file of teacher features sits next to the **`.jpg`**. Runs are **config-driven** and logged to **[MLflow](https://mlflow.org/)** (`train_encoder.py`).

## Layout

| File / directory | Role |
|------------------|------|
| [`train_encoder.py`](train_encoder.py) | Single entrypoint: YAML config, MLflow, optional `torchrun` multi-GPU |
| [`dataset_sa1b.py`](dataset_sa1b.py) | **70% / 10% / 20%** train / val / test split |
| [`configs/tinyvit_baseline.yaml`](configs/tinyvit_baseline.yaml) | Template for Chameleon / Docker (`data.root` = data mount) |
| [`configs/tinyvit_local_sa1b.yaml`](configs/tinyvit_local_sa1b.yaml) | Example paths for a local `MobileSAM-pytorch` sibling checkout |
| [`requirements.txt`](requirements.txt) | Python deps (install **PyTorch** separately for your CUDA / ROCm stack) |
| [`Dockerfile`](Dockerfile) | ROCm 6.0 training image — build from **repository root** |
| [`setup_host.sh`](setup_host.sh) | Chameleon host: install rclone, mount object store read-only (see below) |

**TinyViT import path:** set `mobilesam_root` in YAML, or env `MOBILESAM_ROOT` / `IMMICH_MS_ROOT`, or clone [MobileSAM-pytorch](https://github.com/ChaoningZhang/MobileSAM-pytorch) so `../MobileSAM-pytorch/MobileSAM` exists next to this repo.

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

### 1.3 Secrets and object store mount

On the **server**, create **`~/training.env`** (mode `600`) with at least:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Optional overrides (defaults match `setup_host.sh`): `S3_ENDPOINT_URL`, `OBJSTORE_BUCKET`, `RCLONE_REMOTE_NAME`, `RCLONE_MOUNTPOINT`.

From the cloned repo, run the host prep script (installs fuse/rclone, configures remote, **daemon-mounts** the bucket):

```bash
cd ~/immich-sticker-gen
git pull
bash training/setup_host.sh
```

Confirm the mount (default mountpoint **`/tmp/rclone-tests/object`** unless you override `RCLONE_MOUNTPOINT`):

```bash
ls /tmp/rclone-tests/object | head
```

Point **`data.root`** in your YAML at the directory that **contains** your shard folder (e.g. parent of `sa_000000`) **on that mount** or on local disk.

### 1.4 Clone repo and MobileSAM on the instance

```bash
cd ~
# Replace with your team’s GitHub URL for this repository.
git clone https://github.com/PranjalMinocha/immich-sticker-gen.git
git clone --recursive https://github.com/ChaoningZhang/MobileSAM-pytorch.git
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
   | `data.root` | Directory containing `sa_000000` (or your shard), e.g. rclone mount path + prefix |
   | `output.dir` | Where checkpoints / logs go, e.g. `~/training_out` |
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
python3 train_encoder.py --config /path/to/run.yaml
```

### 3.2 Bare metal (2× GPU, e.g. MI100 ×2)

```bash
cd ~/immich-sticker-gen/training
export MOBILESAM_ROOT=~/MobileSAM-pytorch/MobileSAM
export MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000

torchrun --nproc_per_node=2 train_encoder.py --config /path/to/run.yaml
```

### 3.3 Docker (ROCm) — matches graded “train in container” flow

Build from **repository root**:

```bash
cd ~/immich-sticker-gen
docker build -f training/Dockerfile -t immich-sticker-train:rocm .
```

Example run (adjust host paths for data, MobileSAM, output):

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /tmp/rclone-tests/object:/data:ro \
  -v ~/MobileSAM-pytorch/MobileSAM:/mobilesam:ro \
  -v ~/training_out:/out \
  -e MLFLOW_TRACKING_URI=http://YOUR_MLFLOW_HOST:8000 \
  -e MOBILESAM_ROOT=/mobilesam \
  immich-sticker-train:rocm \
  torchrun --nproc_per_node=2 train_encoder.py --config /out/run.yaml
```

Ensure **`data.root`** inside the YAML matches the **in-container** path (e.g. `/data/...` if you mounted object store at `/data`).

---

## 4. Local development (laptop / workstation)

Not counted for the Chameleon graded runs, but useful for debugging.

```bash
cd training
pip install -r requirements.txt   # plus torch for your platform (CPU/CUDA/ROCm)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # or team server
python train_encoder.py --config configs/tinyvit_local_sa1b.yaml
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
