# Data layout and object storage

## How training reads data today

The training code **does not** load your entire dataset into RAM. It uses PyTorch `DataLoader` workers that **open each `.jpg` / `.npy` (and optional `.json`) on demand** from whatever path you configure—typically an **rclone FUSE mount** of your Chameleon bucket.

That means:

- **Memory**: bounded roughly by batch size × image/embedding tensors, not by total corpus size.
- **Throughput**: random access over a network mount can be **I/O bound** (latency, bandwidth). If workers starve the GPU, reduce `num_workers`, cache hot shards on local NVMe, or stage extracted files to instance scratch.

**Tar archives (`*.tar.gz`) are not read directly.** The current `dataset_sa1b` collectors expect a **directory tree of extracted** `.jpg` files (and parallel `.npy` or split teacher layout). Plan a one-time or periodic **extract / sync** job (script, `make` target, or CI) that materializes:

- Images (+ optional mask JSON) under a path you point `data.root` / `data.image_root` at.
- Teacher embeddings under `data.teacher_root` when using `layout: split_teacher`.

## Layouts

### `colocated` (default)

```
data.root / <shard> / foo.jpg
data.root / <shard> / foo.npy
```

### `split_teacher` (matches Raw-Data + Teacher-Embeddings style buckets)

```yaml
data:
  layout: split_teacher
  image_root: /mount/Raw-Data/extracted
  teacher_root: /mount/Teacher-Embeddings
  shard_dirs: [sa_000000]
```

Requires `image_root/<shard>/<id>.jpg` and `teacher_root/<shard>/<id>.npy` with the **same stem**.

### Masks for full-SAM modes / test IoU

Set `data.annotation_root` and place COCO-style JSON per image (RLE in `annotations[].segmentation` or polygon lists). Resolution:

- `annotation_root/<shard>/<stem>.json`, or
- `annotation_root/<stem>.json`

See `dataset_sa1b.resolve_annotation_json` for lookup order.
