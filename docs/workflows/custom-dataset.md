# Custom dataset visualization

The custom workflow processes a folder-per-sample dataset and writes normalized
inputs separately from generated artifacts.

## Dataset format

```text
examples/custom_dataset/
└── sample_id/
    ├── input.png
    ├── prompt.txt
    └── metadata.json  # optional
```

Supported input extensions are PNG, JPEG, and WebP. Images are converted to RGB,
scaled to the configured maximum side, and aligned to 16-pixel dimensions.

## Commands

```bash
python scripts/run.py --dataset custom --data-dir examples/custom_dataset \
  --selection maze --output-dir output --gpus 0 --dry-run

# Remove --dry-run to infer; use --selection all for every sample.
```

Run `python scripts/run.py --help` for all shared model, dataset, and parallel
options.
