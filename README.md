# immich-sticker-gen

ML feature: sticker generation for [Immich](https://github.com/immich-app/immich) (course project).

## Training (TinyViT encoder distillation)

Encoder distillation against ViT-H teacher embeddings (`.npy` next to each `.jpg`), config-driven with [MLflow](https://mlflow.org/) logging.

**Layout** (in this repo):

- [`training/train_encoder.py`](training/train_encoder.py) — entrypoint
- [`training/dataset_sa1b.py`](training/dataset_sa1b.py) — 70% / 10% / 20% train / val / test split
- [`training/configs/tinyvit_baseline.yaml`](training/configs/tinyvit_baseline.yaml) — template for Chameleon/Docker (`data.root` = mounted data)
- [`training/configs/tinyvit_local_sa1b.yaml`](training/configs/tinyvit_local_sa1b.yaml) — example paths for a local `MobileSAM-pytorch` sibling checkout

**TinyViT source**: set `mobilesam_root` in YAML, or env `MOBILESAM_ROOT`, or place [MobileSAM-pytorch](https://github.com/ChaoningZhang/MobileSAM) next to this repo so `../MobileSAM-pytorch/MobileSAM` exists.

**Local run** (CPU or CUDA/ROCm per your `torch` install):

```bash
cd training
pip install -r requirements.txt  # plus torch for your platform
export MLFLOW_TRACKING_URI=http://129.114.27.248:8000   # or local
python train_encoder.py --config configs/tinyvit_local_sa1b.yaml
```

**Docker (ROCm 6.0)** — build from repo root:

```bash
docker build -f training/Dockerfile -t immich-sticker-train:rocm .
```

**2× GPU** (example):

```bash
cd training
torchrun --nproc_per_node=2 train_encoder.py --config configs/tinyvit_baseline.yaml
```

If the distributed backend fails on AMD, set `train.distributed_backend: gloo` in YAML.

### Chameleon GPU instance (paste into SSH)

Adjust paths to match your machine. This assumes the repo, data (e.g. `sa_000000` with `.jpg` + `.npy`), and MobileSAM `mobile_sam` code are on the instance.

```bash
# --- paths ---
export REPO=~/immich-sticker-gen
export DATA_ROOT=~/MobileSAM-pytorch          # parent of sa_000000
export MS_ROOT=~/MobileSAM-pytorch/MobileSAM # TinyViT package
export OUT=~/training_out
export MLFLOW_TRACKING_URI=http://129.114.27.248:8000

cd "$REPO"
git pull
pip install -r training/requirements.txt
# PyTorch: use your ROCm wheel if not already installed, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

mkdir -p "$OUT"
```

Create a run config (once) by copying the baseline and setting `data.root`:

```bash
cp "$REPO/training/configs/tinyvit_baseline.yaml" "$OUT/run.yaml"
# Edit data.root to $DATA_ROOT (directory that contains sa_000000), and output.dir to $OUT
sed -i "s|root: /data|root: $DATA_ROOT|" "$OUT/run.yaml"
sed -i "s|dir: /tmp/immich_sticker_training_out|dir: $OUT|" "$OUT/run.yaml"
```

**Single GPU:**

```bash
cd "$REPO/training"
export MOBILESAM_ROOT="$MS_ROOT"
python3 train_encoder.py --config "$OUT/run.yaml"
```

**Two GPUs (MI100 x2):**

```bash
cd "$REPO/training"
export MOBILESAM_ROOT="$MS_ROOT"
torchrun --nproc_per_node=2 train_encoder.py --config "$OUT/run.yaml"
```

**Docker (optional):** build from repo root with `docker build -f training/Dockerfile -t immich-sticker-train:rocm .`, mount `$DATA_ROOT`, `$MS_ROOT`, and `$OUT`, set `MLFLOW_TRACKING_URI`, run `torchrun` inside the container.

### How long a run takes

Depends on **number of images**, **epochs**, **batch size**, and **1 vs 2 GPUs**. The tqdm bars show **epoch** progress, **batches per epoch**, and **val/test** passes; use the time for **epoch 1** to estimate total time ≈ `epoch_1_time × epochs`.

Very rough order of magnitude for a **full** `sa_000000`-scale folder and **8 epochs**: often **tens of minutes to a few hours**. A **1-epoch smoke test** (set `train.epochs: 1` in YAML) is usually **minutes**.