# immich-sticker-gen

Complementary ML feature for **[Immich](https://github.com/immich-app/immich)** (self-hosted photo library): **sticker generation** from a user’s photo using interactive prompts (points / box) and a compact segmentation stack derived from SAM / MobileSAM.

This repository is the course project codebase for **ECE-GY 9183** (ML systems design & operations). Components are split by team role (`training/`, `serving/`, etc.); see the [initial implementation rubric](https://ffund.github.io/ml-sys-ops/docs/project.html#initial-implementation-due-apr-6) for deliverables.

## Problem (summary)

Immich users often want quick, high-quality cutouts (“stickers”) from personal photos. The feature uses a **promptable segmentation** model design (SAM-style): **RGB image + user prompts → binary mask → sticker asset**. Training focuses on a **TinyViT** image encoder distilled to match **ViT-H** teacher embeddings, using **SA-1B** lineage data (images + masks; teacher features as `.npy` next to images).

## Repository layout

| Path | Purpose |
|------|---------|
| [`training/`](training/) | Encoder distillation: `train_encoder.py`, YAML configs, Dockerfile, Chameleon host prep (`setup_host.sh`) |
| [`serving/`](serving/) | Serving experiments and notebooks (separate from graded training script path) |

## Training (quick pointer)

All operational steps—Chameleon instance prep, object store mount, editing config, Docker vs bare-metal runs, MLflow—live in **[`training/README.md`](training/README.md)**.

## References

- [Immich](https://github.com/immich-app/immich)
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) / [SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/)
- [MobileSAM](https://arxiv.org/pdf/2306.14289v1) — [MobileSAM-pytorch](https://github.com/ChaoningZhang/MobileSAM-pytorch) (TinyViT + teacher pipeline alignment)
