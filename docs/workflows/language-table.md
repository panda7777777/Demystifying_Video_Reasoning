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

TFRecord extraction additionally requires `tensorflow-datasets` and Pillow. Use
`--tfds-python` to point at a separate environment containing those packages.

## Batch visualization

```bash
python scripts/language_table/visualize.py \
  --data-dir ./data/language-table/0.0.1 \
  --output-root ./output/step_visualization/language_table \
  --tfds-python /path/to/tfds/python \
  --gpus 0,1,2,3 \
  --indices 0:100 \
  --max-denoising-steps 10
```

Use `--dry-run` to inspect deterministic assignments without writing outputs.
Completed episodes are resumed only when their metadata matches the current
`run.json`; otherwise use a new output directory or `--overwrite`.

For a scheduler or manual multi-node launch:

```bash
DATA_DIR=/shared/language-table/0.0.1 \
OUTPUT_ROOT=/shared/results \
NUM_NODES=2 NODE_RANK=0 GPUS=0,1,2,3 \
bash scripts/language_table/launch.sh --indices 0:1000
```

## Results

```bash
python scripts/language_table/results.py --root ./output/run find-prompt "prompt"
python scripts/language_table/results.py --root ./output/run sample 10 --seed 1
python scripts/language_table/results.py --root ./output/run render --workers 8
```

Rendered frames and montages are written under `analysis/{sample_id}/`, leaving
the generated episode artifacts unchanged.
