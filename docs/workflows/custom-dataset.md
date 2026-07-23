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
# Inspect bundled samples.
python scripts/custom/visualize.py --list

# Preview commands without loading a model.
python scripts/custom/visualize.py --sample maze --dry-run

# Process one or all samples.
python scripts/custom/visualize.py --sample maze
python scripts/custom/visualize.py --all
```

Command-line values override `metadata.json`, which overrides workflow defaults.
Use `--overwrite` when replacing an existing result created with different
parameters. Run `python scripts/custom/visualize.py --help` for all options.

Outputs use the common `run.json`, `inputs/`, and `episodes/` structure described
in the repository refactoring plan.
