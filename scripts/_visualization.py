"""Shared helpers for step-level video visualization workflows."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)

LTX_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, "
    "washed out colors, excessive noise, grainy texture, poor lighting, "
    "flickering, motion blur, distorted proportions, artifacts"
)

SUPPORTED_MODELS = ("wan2.2", "wan2.1", "ltx2.3", "vbvr-wan2.2")
WAN_MODELS = frozenset({"wan2.2", "wan2.1", "vbvr-wan2.2"})


@dataclass(frozen=True)
class GenerationConfig:
    """Serializable generation options shared by command-line workflows."""

    model: str = "wan2.2"
    num_frames: int = 49
    num_inference_steps: int = 30
    max_denoising_steps: int | None = None
    seed: int = 1
    fps: int = 16
    negative_prompt: str | None = None

    def validate(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model}")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_denoising_steps is not None:
            if self.model not in WAN_MODELS:
                raise ValueError("max_denoising_steps is supported only for Wan models")
            if not 1 <= self.max_denoising_steps <= self.num_inference_steps:
                raise ValueError(
                    "max_denoising_steps must be between 1 and "
                    "num_inference_steps"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_visualization_steps(spec: str) -> set[int] | None:
    """Parse ``all`` or comma-separated inclusive ranges such as ``0-4,9``."""
    if spec.strip().lower() == "all":
        return None
    steps: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid visualization step range: {token}")
            steps.update(range(start, end + 1))
        else:
            step = int(token)
            if step < 0:
                raise ValueError("Visualization steps cannot be negative")
            steps.add(step)
    if not steps:
        raise ValueError("No visualization steps selected")
    return steps


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def default_negative_prompt(model: str) -> str:
    return LTX_NEGATIVE_PROMPT if model == "ltx2.3" else WAN_NEGATIVE_PROMPT


def make_step_callback(output_dir: Path, fps: int = 16) -> Callable:
    """Create a callback that saves one decoded video per denoising step."""
    from diffsynth.utils.data import save_video

    output_dir.mkdir(parents=True, exist_ok=True)

    def callback(step_index: int, total_steps: int, step_video: Any) -> None:
        path = output_dir / f"step_{step_index:03d}.mp4"
        save_video(step_video, str(path), fps=fps, quality=5)
        print(f"  saved step {step_index + 1}/{total_steps}: {path}", flush=True)

    return callback


def _vram_config(torch: Any) -> dict[str, Any]:
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def build_pipeline(
    model: str,
    *,
    vbvr_model_path: Path | None = None,
    lora_path: str | None = None,
    high_noise_lora_path: str | None = None,
    low_noise_lora_path: str | None = None,
    lora_alpha: float = 1.0,
):
    """Build a supported visualization pipeline."""
    import torch

    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    config = _vram_config(torch)
    if model == "wan2.2":
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            redirect_common_files=False,
            model_configs=[
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    origin_file_pattern=(
                        "high_noise_model/diffusion_pytorch_model*.safetensors"
                    ),
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    origin_file_pattern=(
                        "low_noise_model/diffusion_pytorch_model*.safetensors"
                    ),
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    origin_file_pattern="Wan2.1_VAE.pth",
                    **config,
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="google/umt5-xxl/",
            ),
        )
    elif model == "vbvr-wan2.2":
        if vbvr_model_path is None:
            env_path = os.environ.get("VBVR_MODEL_PATH")
            vbvr_model_path = Path(env_path) if env_path else None
        if vbvr_model_path is None:
            raise ValueError(
                "VBVR-Wan2.2 requires --vbvr-model-path or VBVR_MODEL_PATH"
            )

        def expert(subdirectory: str) -> ModelConfig:
            files = sorted(
                str(path)
                for path in (vbvr_model_path / subdirectory).glob(
                    "diffusion_pytorch_model*.safetensors"
                )
            )
            if not files:
                raise FileNotFoundError(
                    f"No diffusion safetensors found in "
                    f"{vbvr_model_path / subdirectory}"
                )
            return ModelConfig(path=files, **config)

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            redirect_common_files=False,
            model_configs=[
                expert("transformer"),
                expert("transformer_2"),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    origin_file_pattern="Wan2.1_VAE.pth",
                    **config,
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.2-I2V-A14B",
                origin_file_pattern="google/umt5-xxl/",
            ),
        )
    elif model == "wan2.1":
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            redirect_common_files=False,
            model_configs=[
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-I2V-14B-720P",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-I2V-14B-720P",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-I2V-14B-720P",
                    origin_file_pattern="Wan2.1_VAE.pth",
                    **config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-I2V-14B-720P",
                    origin_file_pattern=(
                        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                    ),
                    **config,
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-720P",
                origin_file_pattern="google/umt5-xxl/",
            ),
        )
    elif model == "ltx2.3":
        from diffsynth.pipelines.ltx2_audio_video import (
            LTX2AudioVideoPipeline,
        )

        pipe = LTX2AudioVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    model_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                    origin_file_pattern="model-*.safetensors",
                    **config,
                ),
                *[
                    ModelConfig(
                        model_id="DiffSynth-Studio/LTX-2.3-Repackage",
                        origin_file_pattern=pattern,
                        **config,
                    )
                    for pattern in (
                        "transformer.safetensors",
                        "text_encoder_post_modules.safetensors",
                        "video_vae_decoder.safetensors",
                        "audio_vae_decoder.safetensors",
                        "audio_vocoder.safetensors",
                        "video_vae_encoder.safetensors",
                    )
                ],
            ],
            tokenizer_config=ModelConfig(
                model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"
            ),
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    if model in {"wan2.2", "vbvr-wan2.2"}:
        if high_noise_lora_path:
            pipe.load_lora(pipe.dit, high_noise_lora_path, alpha=lora_alpha)
        if low_noise_lora_path:
            pipe.load_lora(pipe.dit2, low_noise_lora_path, alpha=lora_alpha)
    elif lora_path:
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    return pipe
