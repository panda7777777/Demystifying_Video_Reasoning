# Language-Table visualization

This workflow downloads Language-Table data, extracts the first observation and
instruction, and distributes step-level video generation across GPUs and nodes.
Nodes do not communicate, but must use the same shared dataset and output paths.

## Download

The downloader uses Python's standard library and supports retries and resumable
downloads:

```bash
python scripts/language_table/download.py --list
python scripts/language_table/download.py \
  --datasets language_table \
  --out-dir ./data/language-table
```

Install the Language-Table TFRecord dependencies in the active environment:

```bash
python -m pip install -e '.[language-table]'
```

## Batch visualization

```bash
python scripts/run.py --dataset language-table \
  --data-dir ./data/language-table/0.0.1 \
  --output-dir ./output \
  --gpus 0,1,2,3 \
  --selection 0:100 \
  --split train \
  --max-denoising-steps 10
```

Use `--dry-run` to inspect deterministic assignments without writing outputs.
Completed episodes are resumed only when their metadata matches the current
`run.json`; otherwise use a new output directory or `--overwrite`.

For a scheduler or manual multi-node launch:

```bash
Edit the configuration block in `scripts/launch.sh`, then run it on every node.
Use the same automatically printed `--run-name` on all nodes when launches may
cross a minute boundary.
```

## Results

No post-processing command is required. Every sample contains its initial
frame and prompt, generated video, per-step videos, all per-step frames, and a
paper-ready `output/overview.png`.
