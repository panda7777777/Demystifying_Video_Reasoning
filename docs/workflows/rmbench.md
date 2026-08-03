# RMBench visualization

This workflow reads RMBench's per-episode head-camera videos and global
instructions, extracts the first frame, and distributes step-level video
generation across GPUs and nodes. Source data is never modified.

## Dataset format

Pass the RMBench `data/data` directory. Every selected task must contain
matching videos and instruction files:

```text
data/data/
└── cover_blocks/
    └── demo_clean/
        ├── video/episode0.mp4
        └── instructions/episode0.json
```

The source MP4 is generated from `head_camera/rgb`. The workflow uses its first
frame and the single `seen` instruction by default; pass
`--instruction-split unseen` to select the unseen instruction.

## Batch visualization

Preview a selection without writing files:

```bash
python scripts/run.py --dataset rmbench \
  --data-dir /path/to/RMBench/data/data \
  --output-dir ./output \
  --selection 'cover_blocks,press_button:0:10' \
  --gpus 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` to extract inputs and run inference. The part before the
first colon selects comma-separated tasks and the remainder is a Python-style
episode-position slice.

Wan models save the first 10 denoising steps by default. Override this with
`--max-denoising-steps`. LTX runs all configured steps unless a Wan model is
selected. Use `--extract-only` to prepare inputs without loading a model.
First-frame extraction requires Pillow and `imageio[ffmpeg]`; when those are
installed in a separate environment, pass its interpreter with
`--extraction-python /path/to/python`.

Completed episodes are skipped only when source and generation metadata match.
Use a new output directory or `--overwrite` for changed parameters.

## Multiple nodes

`scripts/launch.sh` exposes all dataset, model, generation, GPU, node,
interpreter, and run-mode options in the user configuration section at the top
of the file. Edit that section explicitly before launching. The remainder of
the script validates the values, changes to the repository root so relative
paths resolve consistently, and constructs the Python command. Its default
interpreter is `/mnt/umm/users/zuojing/env/miniforge3/bin/python`.

For example, configure each node with the same shared paths and:

```bash
TASKS="cover_blocks,press_button"
EPISODES="0:50"
GPUS="0,1,2,3"
NUM_NODES="2"
NODE_RANK="0"  # Change to "1" on the second node.
```

Then run `bash scripts/launch.sh`. Nodes do not communicate, but must
use the same dataset and output paths. Assignments and worker logs are stored
under `runtime/`. Normalized inputs use IDs such as
`cover_blocks__episode_000000` under `samples/`.
