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
  --selection maze --task-name maze --output-dir output --gpus 0 --dry-run

# Remove --dry-run to infer. Selection accepts sample IDs (for example,
# `maze,blocks`) or positional slices (for example, `:10`). Use `all` for every sample.
```

`--task-name` is optional. When provided, it is inserted before the model suffix
in the timestamped run directory; omitting it preserves the original naming.

Validate the configured GPT Image 2 custom datasets, or dry-run/run all of them
serially, with:

```bash
bash scripts/check_gpt_image_2_custom.sh
bash scripts/launch_gpt_image_2_custom.sh --dry-run
bash scripts/launch_gpt_image_2_custom.sh
```

Run `python scripts/run.py --help` for all shared model, dataset, and parallel
options.
