from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_H3_PRESETS: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    "fast": {
        "single": {
            "is_turbo": {
                "sampler": "euler",
                "sigmas": (
                    "1.0000, 0.9950, 0.9825, 0.9607, 0.9234, 0.8553, "
                    "0.7207, 0.4249, 0.3005, 0.0000"
                ),
            },
            "non_turbo": {
                "sampler": "euler",
                "sigmas": (
                    "1.0000, 0.9939, 0.9870, 0.9790, 0.9697, 0.9587, "
                    "0.9455, 0.9293, 0.9091, 0.8831, 0.8485, 0.8000, "
                    "0.6792, 0.4877, 0.2857, 0.0837, 0.0000"
                ),
            },
        },
        "dual": {
            "is_turbo": {
                "sampler": "euler",
                "sigmas": (
                    "1.0000, 0.9950, 0.9825, 0.9607, 0.9234, 0.8553, "
                    "0.7207, 0.4249, 0.3005, 0.0000"
                ),
                "split_step": 4,
            },
            "non_turbo": {
                "sampler": "euler",
                "sigmas": (
                    "1.0000, 0.9939, 0.9870, 0.9790, 0.9697, 0.9587, "
                    "0.9455, 0.9293, 0.9091, 0.8831, 0.8485, 0.8000, "
                    "0.6792, 0.4877, 0.2857, 0.0837, 0.0000"
                ),
                "split_step": 10,
            },
        },
    },
    "medium": {
        "single": {
            "is_turbo": {
                "sampler": "er_sde",
                "sigmas": "1, .9882, .973, .9524, .9231, .878, .8, .6316, .4737, .1579, 0",
            },
            "non_turbo": {
                "sampler": "er_sde",
                "sigmas": (
                    "1.0, 0.9956, 0.9908, 0.9855, 0.9796, 0.973, "
                    "0.9655, 0.9571, 0.9474, 0.9362, 0.9231, 0.9076, "
                    "0.8889, 0.866, 0.8372, 0.8, 0.75, 0.6792, 0.4877, "
                    "0.2857, 0.0837, 0.0"
                ),
            },
        },
        "dual": {
            "is_turbo": {
                "sampler": "er_sde",
                "sigmas": "1, .9882, .973, .9524, .9231, .878, .8, .6316, .4737, .1579, 0",
                "sampler_2": "sa_solver",
                "sigmas_2": ".6316, .4877, .2857, .0837, 0",
            },
            "non_turbo": {
                "sampler": "er_sde",
                "sigmas": (
                    "1.0, 0.9956, 0.9908, 0.9855, 0.9796, 0.973, "
                    "0.9655, 0.9571, 0.9474, 0.9362, 0.9231, 0.9076, "
                    "0.8889, 0.866, 0.8372, 0.8, 0.75, 0.6792, 0.4877, "
                    "0.2857, 0.0837, 0.0"
                ),
                "sampler_2": "sa_solver",
                "sigmas_2": ".6316, .4877, .2857, .0837, 0",
            },
        },
    },
}

_SIGMA_PATTERN = re.compile(r"[-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_h3_sigmas(value: Any, path: str) -> list[float]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty sigma string")
    values = [float(item) for item in _SIGMA_PATTERN.findall(value)]
    if len(values) < 2:
        raise ValueError(f"{path} must contain at least two sigma values")
    if not all(math.isfinite(item) and item >= 0 for item in values):
        raise ValueError(f"{path} must contain finite, non-negative values")
    if any(left < right for left, right in zip(values, values[1:])):
        raise ValueError(f"{path} must be monotonically non-increasing")
    if values[-1] != 0:
        raise ValueError(f"{path} must end with 0")
    return values


def _validate_sampler(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty sampler name")


def _validate_h3_preset_entry(entry: Any, path: str, *, dual: bool) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{path} must be an object")
    _validate_sampler(entry.get("sampler"), f"{path}.sampler")
    sigmas = parse_h3_sigmas(entry.get("sigmas"), f"{path}.sigmas")
    if not dual:
        return

    has_split = "split_step" in entry
    has_second_schedule = "sampler_2" in entry or "sigmas_2" in entry
    if has_split and has_second_schedule:
        raise ValueError(
            f"{path} must use either split_step or sampler_2/sigmas_2, not both"
        )
    if has_split:
        split_step = entry.get("split_step")
        if isinstance(split_step, bool) or not isinstance(split_step, int):
            raise ValueError(f"{path}.split_step must be an integer")
        if split_step <= 0 or split_step >= len(sigmas) - 1:
            raise ValueError(
                f"{path}.split_step must be between 1 and {len(sigmas) - 2}"
            )
        return
    if not has_second_schedule:
        raise ValueError(
            f"{path} must define split_step or sampler_2 and sigmas_2"
        )
    _validate_sampler(entry.get("sampler_2"), f"{path}.sampler_2")
    parse_h3_sigmas(entry.get("sigmas_2"), f"{path}.sigmas_2")


def validate_h3_presets(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("h3_presets.json root must be an object")
    for preset_name in ("fast", "medium"):
        preset = value.get(preset_name)
        if not isinstance(preset, dict):
            raise ValueError(f"h3_presets.json.{preset_name} must be an object")
        for mode_name in ("single", "dual"):
            mode = preset.get(mode_name)
            if not isinstance(mode, dict):
                raise ValueError(
                    f"h3_presets.json.{preset_name}.{mode_name} must be an object"
                )
            for turbo_name in ("is_turbo", "non_turbo"):
                path = f"h3_presets.json.{preset_name}.{mode_name}.{turbo_name}"
                _validate_h3_preset_entry(
                    mode.get(turbo_name),
                    path,
                    dual=mode_name == "dual",
                )
    return value


def load_h3_presets(root: str | Path | None = None) -> dict[str, Any]:
    preset_path = (
        Path(root) / "h3_presets.json"
        if root is not None
        else Path(__file__).resolve().parents[1] / "h3_presets.json"
    )
    if not preset_path.is_file():
        return deepcopy(DEFAULT_H3_PRESETS)
    try:
        value = json.loads(preset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Failed to read valid JSON from {preset_path}: {error}") from error
    try:
        return validate_h3_presets(value)
    except ValueError as error:
        raise ValueError(f"Invalid {preset_path}: {error}") from error


def select_h3_preset(
    presets: dict[str, Any],
    preset_name: str,
    sampling_mode: str,
    is_turbo: bool,
) -> dict[str, Any]:
    if preset_name not in {"fast", "medium"}:
        raise ValueError("sampling_presets must be 'fast' or 'medium'")
    if sampling_mode not in {"single", "dual"}:
        raise ValueError("sampling_mode must be 'single' or 'dual'")
    turbo_name = "is_turbo" if is_turbo else "non_turbo"
    return deepcopy(presets[preset_name][sampling_mode][turbo_name])
