# Training pipeline (overview)

End-to-end flow for Chameleon + Docker + MLflow. Finer behavior is in [`README.md`](README.md) and [`DATA.md`](DATA.md).

## Flowchart

```mermaid
flowchart TB
  subgraph host["GPU host (Chameleon)"]
    ENV["~/training.env\n(AWS keys, optional LOCAL_DATA_ROOT, …)"]
    SH["setup_host.sh"]
    RCLONE["rclone: sync Raw-Data + Teacher-Embeddings\n→ LOCAL_DATA_ROOT"]
    TAR["Extract sa-1b-sample.tar.gz\n→ Raw-Data/extracted/ (once, marker file)"]
    OPTMNT["Optional: RCLONE_ENABLE_MOUNT=1\nFUSE mount whole bucket"]
    ENV --> SH
    SH --> RCLONE --> TAR
    SH -.-> OPTMNT
  end

  subgraph docker["Docker (graded path)"]
    BUILD["docker build -f training/Dockerfile"]
    RUN["docker run: ROCm devices + volumes"]
    VM1["-v LOCAL_DATA_ROOT:/data:ro"]
    VM2["-v MobileSAM:/mobilesam:ro"]
    VM3["-v out dir:/out"]
    BUILD --> RUN
    RUN --> VM1
    RUN --> VM2
    RUN --> VM3
  end

  subgraph train["training/train.py + YAML"]
    CFG["--config *.yaml"]
    MODE{"training.mode"}
    ENC["encoder_distill"]
    FULL["full_sam"]
    CFG --> MODE
    MODE -->|encoder_distill| ENC
    MODE -->|full_sam| FULL
  end

  subgraph dataread["Data layouts"]
    L1["colocated:\nroot/shard/*.jpg + *.npy"]
    L2["split_teacher:\nimage_root + teacher_root"]
  end

  TAR --> dataread
  VM1 --> dataread
  dataread --> MODE

  subgraph mlflow["MLflow"]
    LOG["Params, metrics, system stats\nartifacts: mobile_sam_full.pt, split manifest\nval_previews/ (full_sam, optional)"]
  end
  ENC --> LOG
  FULL --> LOG
```

## Finer details

**Host staging** — `setup_host.sh` sources `~/training.env`, configures rclone for S3-compatible storage, **`rclone sync`**s **`Raw-Data`** and **`Teacher-Embeddings`** into **`LOCAL_DATA_ROOT`** (default `~/training-data`), and extracts **`sa-1b-sample.tar.gz`** into **`Raw-Data/extracted/`** once. Optional whole-bucket FUSE mount (`RCLONE_ENABLE_MOUNT=1`) is for browsing; training I/O should use the synced tree.

**Container** — Image from **`training/Dockerfile`**: Ubuntu, ROCm PyTorch, Python deps. Typical mounts: data at **`/data`**, MobileSAM package at **`/mobilesam`**, writable output (e.g. **`/out`**) for configs and checkpoints.

**Training modes** — Single entrypoint **`train.py`**. **`encoder_distill`**: TinyViT vs teacher **`.npy`**, merge into full SAM, log **`mobile_sam_full.pt`**. **`full_sam`**: mask supervision via **`data.annotation_root`**; same artifact. **Distill → segment**: run **`encoder_distill`**, then **`full_sam`** with **`model.mobile_sam_checkpoint`** set to the first run’s **`mobile_sam_full.pt`**.

**Validation previews (`full_sam`)** — After each epoch’s validation IoU, rank 0 can log **`train.val_preview_samples`** (default **3**) PNGs under **`val_previews/epoch_XXXX/`**: image, **box prompt** (same as training — bbox from GT mask), predicted mask overlay, and GT mask contour. Set **`val_preview_samples: 0`** to disable.

**MLflow** — Tracking URI from environment or YAML; flattened params, metrics, optional test IoU, timing, **psutil** / optional **rocm-smi** system metrics.
