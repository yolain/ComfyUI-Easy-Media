from __future__ import annotations

from collections.abc import Iterable
import importlib.util
from pathlib import Path
import re

try:
    from ..utils.model_memory import cleanup_model_memory
    from ..utils.subtitles import (
        _detect_language,
        _detect_text,
        _restore_subtitle_punctuation,
        _summarize_value,
        normalize_subtitle_segments,
    )
except ImportError:
    from utils.model_memory import cleanup_model_memory
    from utils.subtitles import (
        _detect_language,
        _detect_text,
        _restore_subtitle_punctuation,
        _summarize_value,
        normalize_subtitle_segments,
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def missing_dependencies() -> list[str]:
    missing: list[str] = []
    if not _module_available("torch"):
        missing.append("torch")
    if not _module_available("whisper"):
        missing.append("openai-whisper")
    return missing


def _count_indexed_layers(keys: Iterable[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    return len({int(match.group(1)) for key in keys if (match := pattern.match(key))})


def _infer_whisper_head_count(state_size: int) -> int:
    if state_size % 64 == 0:
        return state_size // 64
    known = {
        384: 6,
        512: 8,
        768: 12,
        1024: 16,
        1280: 20,
    }
    if state_size in known:
        return known[state_size]
    raise RuntimeError(f"Cannot infer Whisper attention head count from state size {state_size}")


def _replace_hf_whisper_block_key(key: str) -> str:
    replacements = (
        (".self_attn.q_proj.", ".attn.query."),
        (".self_attn.k_proj.", ".attn.key."),
        (".self_attn.v_proj.", ".attn.value."),
        (".self_attn.out_proj.", ".attn.out."),
        (".encoder_attn.q_proj.", ".cross_attn.query."),
        (".encoder_attn.k_proj.", ".cross_attn.key."),
        (".encoder_attn.v_proj.", ".cross_attn.value."),
        (".encoder_attn.out_proj.", ".cross_attn.out."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".fc1.", ".mlp.0."),
        (".fc2.", ".mlp.2."),
        (".final_layer_norm.", ".mlp_ln."),
    )
    value = key
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _openai_whisper_key_from_hf(key: str) -> str | None:
    direct_replacements = {
        "model.encoder.embed_positions.weight": "encoder.positional_embedding",
        "model.encoder.layer_norm.weight": "encoder.ln_post.weight",
        "model.encoder.layer_norm.bias": "encoder.ln_post.bias",
        "model.decoder.embed_tokens.weight": "decoder.token_embedding.weight",
        "model.decoder.embed_positions.weight": "decoder.positional_embedding",
        "model.decoder.layer_norm.weight": "decoder.ln.weight",
        "model.decoder.layer_norm.bias": "decoder.ln.bias",
    }
    if key in direct_replacements:
        return direct_replacements[key]
    if key.startswith("model.encoder.conv"):
        return key.replace("model.encoder.", "encoder.", 1)
    if key.startswith("model.encoder.layers."):
        return _replace_hf_whisper_block_key(key.replace("model.encoder.layers.", "encoder.blocks.", 1))
    if key.startswith("model.decoder.layers."):
        return _replace_hf_whisper_block_key(key.replace("model.decoder.layers.", "decoder.blocks.", 1))
    if key.startswith("proj_out."):
        return None
    return None


def _convert_hf_whisper_state_dict(sd: dict[str, object]) -> tuple[dict[str, object], object]:
    from whisper.model import ModelDimensions  # type: ignore[import]

    required_keys = (
        "model.encoder.conv1.weight",
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
    )
    missing_keys = [key for key in required_keys if key not in sd]
    if missing_keys:
        raise RuntimeError(f"Whisper safetensors file is missing required weights: {', '.join(missing_keys)}")

    encoder_conv1 = sd["model.encoder.conv1.weight"]
    encoder_positions = sd["model.encoder.embed_positions.weight"]
    decoder_tokens = sd["model.decoder.embed_tokens.weight"]
    decoder_positions = sd["model.decoder.embed_positions.weight"]
    n_audio_state = int(encoder_conv1.shape[0])  # type: ignore[attr-defined]
    n_mels = int(encoder_conv1.shape[1])  # type: ignore[attr-defined]
    n_audio_ctx = int(encoder_positions.shape[0])  # type: ignore[attr-defined]
    n_vocab = int(decoder_tokens.shape[0])  # type: ignore[attr-defined]
    n_text_state = int(decoder_tokens.shape[1])  # type: ignore[attr-defined]
    n_text_ctx = int(decoder_positions.shape[0])  # type: ignore[attr-defined]
    keys = tuple(sd.keys())
    dims = ModelDimensions(
        n_mels=n_mels,
        n_audio_ctx=n_audio_ctx,
        n_audio_state=n_audio_state,
        n_audio_head=_infer_whisper_head_count(n_audio_state),
        n_audio_layer=_count_indexed_layers(keys, "model.encoder.layers"),
        n_vocab=n_vocab,
        n_text_ctx=n_text_ctx,
        n_text_state=n_text_state,
        n_text_head=_infer_whisper_head_count(n_text_state),
        n_text_layer=_count_indexed_layers(keys, "model.decoder.layers"),
    )

    converted: dict[str, object] = {}
    unexpected: list[str] = []
    for key, value in sd.items():
        next_key = _openai_whisper_key_from_hf(key)
        if next_key is None:
            if not key.startswith("proj_out."):
                unexpected.append(key)
            continue
        converted[next_key] = value
    if unexpected:
        raise RuntimeError(
            "Whisper safetensors file contains unsupported weights: "
            + ", ".join(unexpected[:8])
            + ("..." if len(unexpected) > 8 else "")
        )
    return converted, dims


def _load_openai_whisper_from_audio_encoder(model_path: Path, device: str) -> object:
    import comfy.utils
    from whisper.model import Whisper  # type: ignore[import]

    state_dict = comfy.utils.load_torch_file(str(model_path), safe_load=True)
    converted_state_dict, dims = _convert_hf_whisper_state_dict(state_dict)
    model = Whisper(dims)
    missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:8]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:8]}")
        raise RuntimeError("Failed to load Whisper safetensors weights: " + "; ".join(details))
    return model.to(device)


def recognize_subtitle_segments(audio_path: Path, model_path: Path) -> list[dict]:
    import torch

    import whisper  # type: ignore[import]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    try:
        model = _load_openai_whisper_from_audio_encoder(model_path, device)
        result = model.transcribe(
            str(audio_path),
            verbose=False,
            word_timestamps=False,
        )
        transcript = _detect_text(result)
        detected_language = _detect_language(result)
        segments = normalize_subtitle_segments(result)
        if segments:
            return _restore_subtitle_punctuation(segments, transcript, detected_language)
        raise RuntimeError(
            "Whisper Large V3 did not return timestamped subtitle text. "
            f"transcribe={_summarize_value(result)}"
        )
    finally:
        cleanup_model_memory(model)
