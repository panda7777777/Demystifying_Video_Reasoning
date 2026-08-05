"""Shared helpers for step-level video visualization workflows."""

from __future__ import annotations

import json
import os
import re
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

SUPPORTED_MODEL_FAMILIES = ("wan2.2", "wan2.1", "ltx2.3", "vbvr-wan2.2")
WAN_MODELS = frozenset({"wan2.2", "wan2.1", "vbvr-wan2.2"})
HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_model_source(model: str) -> str:
    """Validate and normalize an absolute directory or Hugging Face repo id."""
    path = Path(model)
    if path.is_absolute():
        if not path.is_dir():
            raise ValueError(f"Model path is not an existing directory: {model}")
        return str(path.resolve())
    if not HF_REPO_ID.fullmatch(model):
        raise ValueError(
            "model must be an absolute directory or a Hugging Face repository "
            "id such as Wan-AI/Wan2.2-I2V-A14B"
        )
    return model


def model_family(model: str) -> str:
    """Infer the supported pipeline family from a model source."""
    name = model.rstrip("/").rsplit("/", 1)[-1].lower()
    if "vbvr" in name and "wan2.2" in name:
        return "vbvr-wan2.2"
    if "wan2.2" in name:
        return "wan2.2"
    if "wan2.1" in name:
        return "wan2.1"
    if "ltx-2.3" in name or "ltx2.3" in name:
        return "ltx2.3"

    path = Path(model)
    if path.is_absolute():
        if (path / "transformer").is_dir() and (path / "transformer_2").is_dir():
            return "vbvr-wan2.2"
        if (path / "high_noise_model").is_dir() and (path / "low_noise_model").is_dir():
            return "wan2.2"
        if (path / "transformer.safetensors").is_file():
            return "ltx2.3"
        if list(path.glob("diffusion_pytorch_model*.safetensors")):
            return "wan2.1"
    raise ValueError(
        f"Cannot infer a supported model family from {model!r}; supported families: "
        + ", ".join(SUPPORTED_MODEL_FAMILIES)
    )


@dataclass(frozen=True)
class GenerationConfig:
    """Serializable generation options shared by command-line workflows."""

    model: str = "Wan-AI/Wan2.2-I2V-A14B"
    num_frames: int = 49
    num_inference_steps: int = 30
    max_denoising_steps: int | None = None
    seed: int = 1
    fps: int = 16
    negative_prompt: str | None = None

    def validate(self) -> None:
        validate_model_source(self.model)
        family = model_family(self.model)
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_denoising_steps is not None:
            if family not in WAN_MODELS:
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
    return LTX_NEGATIVE_PROMPT if model_family(model) == "ltx2.3" else WAN_NEGATIVE_PROMPT


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


def _load_wan_adapter(pipe: Any, dit: Any, path: str, alpha: float) -> None:
    """Load a regular LoRA or auto-detected Skip-LoRA checkpoint."""
    import re

    import torch

    from diffsynth.core import load_state_dict

    state_dict = load_state_dict(
        path, torch_dtype=getattr(pipe, "torch_dtype", torch.bfloat16), device="cpu"
    )
    # Accelerate/PEFT training checkpoints save keys relative to the complete
    # training object.  In inference we load into one DiT at a time, so remove
    # that wrapper prefix before handing the weights to DiffSynth's LoRA loader.
    for prefix in ("pipe.dit.", "module.pipe.dit."):
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key.removeprefix(prefix): value for key, value in state_dict.items()
            }
            break
    if not any(key.endswith(".skip_proj.weight") for key in state_dict):
        pipe.load_lora(dit, state_dict=state_dict, alpha=alpha)
        return

    from diffsynth.core.vram.layers import AutoWrappedNonRecurseModule
    from diffsynth.diffusion.skip_lora import (
        SkipLoRAConfig,
        SkipLoRALinear,
        attach_skip_lora,
    )

    lora_a = next(
        (value for key, value in state_dict.items() if key.endswith(".lora_A.weight")),
        None,
    )
    if lora_a is None:
        raise ValueError(f"Skip-LoRA checkpoint has no lora_A weights: {path}")
    rank = lora_a.shape[0]
    config = SkipLoRAConfig(
        rank=rank,
        lora_alpha=rank * alpha,
        combine_mode="linear",
        carry_mode="accumulate",
        normalize_skip=False,
        detach_across_blocks=True,
        detach_within_block=True,
    )
    attach_skip_lora(dit, config)

    # Skip-LoRA parameters stay resident while the frozen block weights are
    # managed by CPU offload.
    target_device = getattr(pipe, "device", "cuda")
    target_dtype = getattr(pipe, "torch_dtype", torch.bfloat16)
    for module in dit.modules():
        if isinstance(module, SkipLoRALinear):
            for name, parameter in module.named_parameters(recurse=True):
                if not name.startswith("base."):
                    parameter.data = parameter.data.to(
                        device=target_device, dtype=target_dtype
                    )

    if (
        len(dit.blocks)
        and isinstance(dit.blocks[0], AutoWrappedNonRecurseModule)
    ):
        pattern = re.compile(r"^(blocks\.\d+)\.(?!module\.)(.+)$")
        state_dict = {
            (f"{match.group(1)}.module.{match.group(2)}" if (match := pattern.match(key)) else key): value
            for key, value in state_dict.items()
        }

    result = dit.load_state_dict(state_dict, strict=False)
    adapter_markers = ("lora_A.weight", "lora_B.weight", "skip_proj.weight")
    missing = [
        key for key in result.missing_keys if any(marker in key for marker in adapter_markers)
    ]
    if result.unexpected_keys or missing:
        raise RuntimeError(
            "Skip-LoRA checkpoint does not match the Wan DiT: "
            f"unexpected={result.unexpected_keys[:5]}, missing={missing[:5]}"
        )
    print(
        f"[Skip-LoRA] loaded {len(state_dict)} tensors from {path} "
        f"(rank={rank}, scale={alpha}, combine=linear, carry=accumulate)",
        flush=True,
    )


def build_pipeline(
    model: str,
    *,
    lora_path: str | None = None,
    high_noise_lora_path: str | None = None,
    low_noise_lora_path: str | None = None,
    lora_alpha: float = 1.0,
):
    """Build a supported visualization pipeline."""
    import torch

    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    model = validate_model_source(model)
    family = model_family(model)
    config = _vram_config(torch)

    def source_config(
        pattern: str | None = None,
        *,
        source: str = model,
        with_vram: bool = True,
    ) -> ModelConfig:
        options = config if with_vram else {}
        path = Path(source)
        if path.is_absolute():
            if pattern is None:
                matched: str | list[str] = str(path)
            else:
                files = sorted(str(item) for item in path.glob(pattern))
                if not files:
                    raise FileNotFoundError(f"No model files matching {path / pattern}")
                matched = files
            return ModelConfig(path=matched, **options)
        return ModelConfig(model_id=source, origin_file_pattern=pattern, **options)

    if family == "wan2.2":
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            redirect_common_files=False,
            model_configs=[
                source_config("high_noise_model/diffusion_pytorch_model*.safetensors"),
                source_config("low_noise_model/diffusion_pytorch_model*.safetensors"),
                source_config("models_t5_umt5-xxl-enc-bf16.pth"),
                source_config("Wan2.1_VAE.pth"),
            ],
            tokenizer_config=source_config("google/umt5-xxl/", with_vram=False),
        )
    elif family == "vbvr-wan2.2":
        base_source = "Wan-AI/Wan2.2-I2V-A14B"
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            redirect_common_files=False,
            model_configs=[
                source_config("transformer/diffusion_pytorch_model*.safetensors"),
                source_config("transformer_2/diffusion_pytorch_model*.safetensors"),
                source_config("models_t5_umt5-xxl-enc-bf16.pth", source=base_source),
                source_config("Wan2.1_VAE.pth", source=base_source),
            ],
            tokenizer_config=source_config("google/umt5-xxl/", source=base_source, with_vram=False),
        )
    elif family == "wan2.1":
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            redirect_common_files=False,
            model_configs=[
                source_config("diffusion_pytorch_model*.safetensors"),
                source_config("models_t5_umt5-xxl-enc-bf16.pth"),
                source_config("Wan2.1_VAE.pth"),
                source_config("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ],
            tokenizer_config=source_config("google/umt5-xxl/", with_vram=False),
        )
    elif family == "ltx2.3":
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
                    source_config(pattern)
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

    if family in {"wan2.2", "vbvr-wan2.2"}:
        if high_noise_lora_path:
            _load_wan_adapter(pipe, pipe.dit, high_noise_lora_path, lora_alpha)
        if low_noise_lora_path:
            _load_wan_adapter(pipe, pipe.dit2, low_noise_lora_path, lora_alpha)
    elif lora_path:
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    return pipe
