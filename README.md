<div align="center">
    
# Demystifying Video Reasoning

</div>
<div align="center">
    
<p align="center">
    <a href="https://www.wruisi.com/demystifying_video_reasoning" target="_blank">
        <img alt="Homepage" src="https://img.shields.io/badge/Project%20-%20Homepage-4285F4" height="60" />
    </a>
    <a href="https://arxiv.org/abs/2603.16870" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-PDF-red?logo=arxiv" height="60" />
    </a>
    <a href="https://huggingface.co/papers/2603.16870" target="_blank">
        <img alt="HuggingFace" src="https://img.shields.io/badge/HuggingFace-Paper-orange" height="60" />
    </a>
    <a href="https://www.youtube.com/watch?v=Gs9TPZmzo-s" target="_blank">
        <img alt="Video" src="https://img.shields.io/badge/YouTube-Video-FF0000?logo=YouTube&logoColor=white" height="60" />
    </a>
</p>

</div>

## Overview

[![Watch the video](https://img.youtube.com/vi/Gs9TPZmzo-s/maxresdefault.jpg)](https://www.youtube.com/watch?v=Gs9TPZmzo-s)

This is the official repository for **[Demystifying Video Reasoning](https://huggingface.co/papers/2603.16870)**.

## Introduction
Recent advances in video generation have revealed an unexpected phenomenon: diffusion-based video models exhibit non-trivial reasoning capabilities. Prior work attributes this to a Chain-of-Frames (CoF) mechanism, where reasoning is assumed to unfold sequentially across video frames. In this work, we challenge this assumption and uncover a fundamentally different mechanism. We show that reasoning in video models instead primarily emerges along the diffusion denoising steps. Through qualitative analysis and targeted probing experiments, we find that models explore multiple candidate solutions in early denoising steps and progressively converge to a final answer, a process we term **Chain-of-Steps (CoS)**. Beyond this core mechanism, we identify several emergent reasoning behaviors critical to model performance: (1) **working memory**, enabling persistent reference; (2) **self-correction and enhancement**, allowing recovery from incorrect intermediate solutions; and (3) **perception before action**, where early steps establish semantic grounding and later steps perform structured manipulation. During a diffusion step, we further uncover self-evolved **functional specialization** within Diffusion Transformers, where early layers encode dense perceptual structure, middle layers execute reasoning, and later layers consolidate latent representations. Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds. Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence.

## News
- [2026-06-18] We are proud to share that our paper has been accepted to ECCV 2026!
- [2026-04-15] Tools for intermediate step decoding and layer-wise token-level visualization are released.
- [2026-03-20] We have released the paper [Demystifying Video Reasoning](https://huggingface.co/papers/2603.16870).

## Release Plan

- [x] **Intermediate steps decoding tool**
- [x] **Layer-wise token-Level visualization tool**

## Installation

### Visualization results dashboard

Launch the offline dashboard for the latest Language-Table visualization run:

```bash
python tools/results_dashboard.py
```

Open `http://<server-ip>:8000` in a browser. Use `A`/`D`, the left/right arrow
keys, or enter a page number to navigate. The default five samples per page
can be changed from the UI or at launch time:

```bash
python tools/results_dashboard.py \
  --result-dir output/20260804_1513_wan22_ltable \
  --page-size 8 \
  --host 0.0.0.0 \
  --port 8000
```

The server uses only the Python standard library. It dynamically discovers
available denoising steps and serves only media from the selected result
directory.

### Python environment

```bash
pip install -e .
pip install matplotlib scikit-learn   # for token visualization
```

## Model Download

#### Wan2.2-I2V-A14B

Download from [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B):

```bash
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./models/Wan-AI/Wan2.2-I2V-A14B
```

#### Wan2.1-I2V-14B-720P

Download from [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P):

```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan-AI/Wan2.1-I2V-14B-720P
```

#### LTX-2.3 (Repackaged)

Download from [Hugging Face](https://huggingface.co/DiffSynth-Studio/LTX-2.3-Repackage):

```bash
huggingface-cli download DiffSynth-Studio/LTX-2.3-Repackage --local-dir ./models/DiffSynth-Studio/LTX-2.3-Repackage
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir ./models/google/gemma-3-12b-it-qat-q4_0-unquantized
```

> **Note:** Models are also auto-downloaded at runtime if not found locally.

#### VBVR lora model family trained on [VBVR-Dataset](https://huggingface.co/datasets/Video-Reason/VBVR-Dataset)

Download VBVR-Wan2.2 from [Hugging Face](https://huggingface.co/Video-Reason/VBVR-Wan2.2-diffsynth), VBVR-Wan2.1 from [Hugging Face](https://huggingface.co/Video-Reason/VBVR-Wan2.1-diffsynth), VBVR-LTX2.3 from [Hugging Face](https://huggingface.co/Video-Reason/VBVR-LTX2.3-diffsynth):
```bash
huggingface-cli download Video-Reason/VBVR-Wan2.1-diffsynth --local-dir ./models/VBVR/VBVR-Wan2.1-diffsynth
huggingface-cli download Video-Reason/VBVR-Wan2.2-diffsynth --local-dir ./models/VBVR/VBVR-Wan2.2-diffsynth
huggingface-cli download Video-Reason/VBVR-LTX2.3-diffsynth --local-dir ./models/VBVR/VBVR-LTX2.3-diffsynth
```

#### Large Video Planner (LVP)

LVP uses the Wan2.1 I2V 14B 480P components plus its full fine-tuned
Lightning checkpoint. Both paths are supplied explicitly, so the 66 GB LVP
checkpoint is memory-mapped and no weight conversion or copy is required:

```bash
python scripts/run.py \
  --model large-video-planner \
  --lvp-base-model /mnt/umm/users/zuojing/models/Wan2.1-I2V-14B-480P \
  --lvp-checkpoint /mnt/umm/users/zuojing/models/LVP/checkpoints/lvp_14B.ckpt \
  --dataset custom \
  --data-dir examples/custom_dataset \
  --gpus 0
```

`lvp` is accepted as a shorter model alias. The base directory must contain
the Wan diffusion shards, T5, VAE, CLIP, and `google/umt5-xxl` tokenizer.

## Evaluation Data

Download the VBVR-Bench evaluation data from Hugging Face:

```bash
huggingface-cli download Video-Reason/VBVR-Bench-Data --repo-type dataset --local-dir ./data/VBVR-Bench
```

The evaluation data has the following structure:

```
data/VBVR-Bench/
├── In-Domain_50/
│   ├── G-xxx_task_name_data-generator/
│   │   ├── 00000/
│   │   │   ├── first_frame.png
│   │   │   ├── final_frame.png
│   │   │   ├── ground_truth.mp4
│   │   │   └── prompt.txt
│   │   ├── 00001/
│   │   └── ...
│   └── ...
└── Out-of-Domain_50/
    └── ...
```

## Tools

### Compose intermediate steps into a GIF

Create a publication-style animated strip with a frozen initial frame, five
uniformly sampled intermediate-step videos, and a centered input prompt:

```bash
python tools/compose_step_gif.py \
  --videos output/sample/steps/*.mp4 \
  --initial-image output/sample/input/initial_frame.png \
  --prompt-file output/sample/input/prompt.txt \
  --num-steps 5 \
  --output output/sample/step_progression.gif
```

The initial image is optional; without it, the first frame of the earliest
selected video is frozen. Videos are naturally sorted (`step_2` before
`step_10`), sampled at equal intervals including both endpoints, synchronized
to the shortest clip, and letterboxed to preserve aspect ratio. Use `--fps`
and `--panel-width` to trade file size for temporal or spatial resolution.

For an existing experiment run, batch-process selected dataset indices with:

```bash
python tools/batch_compose_step_gifs.py \
  --result-dir output/20260807_1527_wan22_ltable \
  --indices '37,998,11,22,47,985,986,987,997,8,29,41,45,1,14,15,39,54,999'
```

The script resolves each requested value against `source.index` in the sample
metadata. GIFs are written to `<result-dir>/visualizations/step_gifs` by
default. The compact layout hides panel titles, uses the bundled Times New
Roman Bold font in `assets/fonts`, and leaves three pixels between panels.
Use `--font-path /path/to/timesbd.ttf` to override it. Add `--show-labels` when
step titles are needed. Existing files are skipped unless `--overwrite` is
supplied. Prompt text scales with panel resolution (24 px at the default
256-pixel panel width, or 60 px with `--panel-width 640`).

### 1. Per-Diffusion-Step Visualization (VBVR-Bench)

Saves intermediate video snapshots at selected denoising steps, showing how generation evolves from noise to the final output. Automatically processes all splits and tasks in VBVR-Bench.

```bash
# Wan2.2 base model
python tools/step_visualization.py \
    --model wan2.2 \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/step_viz/wan2.2

# Wan2.2 with LoRA
python tools/step_visualization.py \
    --model wan2.2 \
    --high_noise_lora_path ./models/VBVR/VBVR-Wan2.2-diffsynth/high_noise_lora.safetensors \
    --low_noise_lora_path ./models/VBVR/VBVR-Wan2.2-diffsynth/low_noise_lora.safetensors \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/step_viz/wan2.2_lora

# Wan2.1 base model
python tools/step_visualization.py \
    --model wan2.1 \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/step_viz/wan2.1

# Wan2.1 with LoRA
python tools/step_visualization.py \
    --model wan2.1 \
    --lora_path ./models/VBVR/VBVR-Wan2.1-diffsynth/lora.safetensors \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/step_viz/wan2.1_lora

# LTX2.3
python tools/step_visualization.py \
    --model ltx2.3 \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/step_viz/ltx2.3 \
    --num_inference_steps 40

# LTX2.3 with LoRA
python tools/step_visualization.py \
    --model ltx2.3 \
    --lora_path ./models/VBVR/VBVR-LTX2.3-diffsynth/lora.safetensors \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/step_viz/ltx2.3_lora \
    --num_inference_steps 40
```

### 2. Token-Level Feature Map Visualization (VBVR-Bench)

Hooks into DiT transformer blocks to capture hidden states, producing spatial heatmaps, PCA maps, temporal energy curves, and cross-step summary plots. Automatically processes all splits and tasks in VBVR-Bench.

```bash
python tools/token_visualization.py \
    --model wan2.2 \
    --eval_root ./data/VBVR-Bench \
    --output_root ./output/token_viz/wan2.2 \
    --layers 0 10 20 30 39 \
    --max_samples 5
```

### Unified Dataset Inference

All custom, RMBench, and Language-Table runs now share one entry point and one
result format. The output argument names only a parent directory; a run folder
such as `20260802_1530_wan22_rmbench` is created automatically.

```bash
python scripts/run.py \
    --model wan2.2 \
    --dataset rmbench \
    --data-dir /path/to/RMBench/data/data \
    --output-dir ./output \
    --selection 'cover_blocks,press_button:0:10' \
    --gpus 0,1,2,3
```

For the editable launcher, set the important values at the top of
`scripts/launch.sh`, then run `bash scripts/launch.sh`.

### 3. Custom Input - Step Visualization

Run step visualization on a single image + text prompt (no VBVR-Bench data needed):

```bash
python tools/custom_step_visualization.py \
    --model wan2.2 \
    --image ./my_image.png \
    --prompt "A cat walks across the room" \
    --output-dir ./output/custom_step \
    --num-frames 81
```

### 4. Custom Input - Token Visualization

Run token-level feature map visualization on a single image + text prompt (no VBVR-Bench data needed):

```bash
python tools/custom_token_visualization.py \
    --model wan2.2 \
    --image ./my_image.png \
    --prompt "A cat walks across the room" \
    --output_dir ./output/custom_token \
    --num_frames 81 \
    --layers 0 10 20 30 39
```

### 5. Dataset Workflows

Folder-based custom inputs and distributed Language-Table and RMBench
processing are available under `scripts/`. See
[Custom dataset workflow](docs/workflows/custom-dataset.md) and
[Language-Table workflow](docs/workflows/language-table.md), or the
[RMBench workflow](docs/workflows/rmbench.md).

## Common Options

| Argument | Description | Default |
|---|---|---|
| `--model` | Model family: `wan2.2`, `wan2.1`, `ltx2.3` | required |
| `--lora_path` | LoRA weights (wan2.1 / ltx2.3) | None |
| `--high_noise_lora_path` | High-noise DiT LoRA (wan2.2) | None |
| `--low_noise_lora_path` | Low-noise DiT LoRA (wan2.2) | None |
| `--lora_alpha` | LoRA merge alpha | 1.0 |
| `--num_inference_steps` | Denoising steps | 50 |
| `--seed` | Random seed | 1 |
| `--vis_steps` | Steps to visualize: `all` or `0-19,45-49` | all |
| `--max_samples` | Max samples per task (bench tools) | None (all) |

## Citation

```bibtex
@article{wang2026demystifing,
  title={Demystifing Video Reasoning},
  author={Wang, Ruisi and Cai, Zhongang and Pu, Fanyi and Xu, Junxiang and Yin, Wanqi and Wang, Maijunxian and Ji, Ran and Gu, Chenyang and Li, Bo and Huang, Ziqi and Deng, Hokin and Lin, Dahua and Liu, Ziwei and Yang, Lei},
  journal={arXiv preprint arXiv:2603.16870},
  year={2026}
}
```

## Acknowledgements

This project includes code that is modified from the original work by the DiffSynth-Studio team.

* Source repository: https://github.com/modelscope/DiffSynth-Studio
* Original project: **modelscope/DiffSynth-Studio**

We gratefully acknowledge the authors and contributors of DiffSynth-Studio for their work.
Please refer to the original repository for full details, updates, and licensing information.
