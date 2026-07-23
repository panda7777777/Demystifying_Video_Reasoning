#!/usr/bin/env python3
"""Generate a video and decoded predictions for selected denoising steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._visualization import (  # noqa: E402
    GenerationConfig,
    SUPPORTED_MODELS,
    atomic_write_json,
    build_pipeline,
    default_negative_prompt,
    make_step_callback,
    parse_visualization_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        help="Maximum number of denoising steps to execute (Wan only)",
    )
    parser.add_argument(
        "--visualization-steps",
        default="all",
        help="'all' or comma-separated steps/ranges such as '0-9,20-24'",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--vbvr-model-path", type=Path)
    parser.add_argument("--lora-path")
    parser.add_argument("--high-noise-lora-path")
    parser.add_argument("--low-noise-lora-path")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--save-noise-schedule", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GenerationConfig(
        model=args.model,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        max_denoising_steps=args.max_denoising_steps,
        seed=args.seed,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )
    try:
        config.validate()
        visualization_steps = parse_visualization_steps(
            args.visualization_steps
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if not args.image.is_file():
        raise SystemExit(f"error: input image not found: {args.image}")
    if not args.prompt.strip():
        raise SystemExit("error: prompt cannot be empty")

    image = Image.open(args.image).convert("RGB")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(
        args.model,
        vbvr_model_path=args.vbvr_model_path,
        lora_path=args.lora_path,
        high_noise_lora_path=args.high_noise_lora_path,
        low_noise_lora_path=args.low_noise_lora_path,
        lora_alpha=args.lora_alpha,
    )

    if args.save_noise_schedule and args.model != "ltx2.3":
        pipe.scheduler.set_timesteps(args.num_inference_steps, shift=5.0)
        schedule = [
            {
                "step": index,
                "timestep": round(timestep.item(), 4),
                "sigma": round(sigma.item(), 6),
            }
            for index, (sigma, timestep) in enumerate(
                zip(pipe.scheduler.sigmas, pipe.scheduler.timesteps)
            )
        ]
        atomic_write_json(args.output_dir / "noise_schedule.json", {"steps": schedule})

    negative_prompt = args.negative_prompt or default_negative_prompt(args.model)
    callback = make_step_callback(args.output_dir / "steps", fps=args.fps)
    output_path = args.output_dir / "generated.mp4"

    if args.model == "ltx2.3":
        from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

        video, audio = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            input_images=[image],
            input_images_indexes=[0],
            input_images_strength=1.0,
            num_frames=args.num_frames,
            seed=args.seed,
            tiled=True,
            height=image.height,
            width=image.width,
            num_inference_steps=args.num_inference_steps,
            step_callback=callback,
            vis_steps=visualization_steps,
        )
        write_video_audio_ltx2(
            video=video,
            audio=audio,
            output_path=str(output_path),
            fps=args.fps,
            audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
        )
    else:
        from diffsynth.utils.data import save_video

        video = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            input_image=image,
            num_frames=args.num_frames,
            seed=args.seed,
            tiled=True,
            height=image.height,
            width=image.width,
            num_inference_steps=args.num_inference_steps,
            step_callback=callback,
            vis_steps=visualization_steps,
            max_denoising_steps=args.max_denoising_steps,
        )
        save_video(video, str(output_path), fps=args.fps, quality=5)

    atomic_write_json(
        args.output_dir / "metadata.json",
        {
            "generation": config.to_dict(),
            "image": str(args.image.resolve()),
            "prompt": args.prompt,
            "visualization_steps": args.visualization_steps,
        },
    )
    print(f"Generated video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
