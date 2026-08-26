import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.h3_presets import (  # noqa: E402
    DEFAULT_H3_PRESETS,
    load_h3_presets,
    select_h3_preset,
)


def test_load_h3_presets_uses_embedded_defaults_without_user_file(tmp_path):
    result = load_h3_presets(tmp_path)

    assert result == DEFAULT_H3_PRESETS
    assert result is not DEFAULT_H3_PRESETS
    assert result["light"]["dual"]["is_turbo"]["split_step"] == 4


def test_load_h3_presets_reads_valid_user_override(tmp_path):
    custom = json.loads(json.dumps(DEFAULT_H3_PRESETS))
    custom["light"]["single"]["is_turbo"]["sampler"] = "heun"
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    (preset_dir / "h3_sample.json").write_text(json.dumps(custom))

    result = load_h3_presets(tmp_path)

    assert result["light"]["single"]["is_turbo"]["sampler"] == "heun"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("medium"),
        lambda value: value["light"]["single"]["is_turbo"].update(
            sigmas="1, 0.5"
        ),
        lambda value: value["light"]["dual"]["is_turbo"].update(
            split_step=99
        ),
        lambda value: value["medium"]["dual"]["is_turbo"].pop(
            "sampler_2"
        ),
    ],
)
def test_load_h3_presets_rejects_invalid_user_configuration(tmp_path, mutate):
    custom = json.loads(json.dumps(DEFAULT_H3_PRESETS))
    mutate(custom)
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    (preset_dir / "h3_sample.json").write_text(json.dumps(custom))

    with pytest.raises(ValueError, match="Invalid .*h3_sample.json"):
        load_h3_presets(tmp_path)


def test_select_h3_preset_uses_json_dual_turbo_branch():
    result = select_h3_preset(DEFAULT_H3_PRESETS, "medium", "dual", True)

    assert result["sampler"] == "er_sde"
    assert result["sampler_2"] == "sa_solver"


def test_select_h3_preset_uses_non_turbo_for_unknown_detection():
    result = select_h3_preset(DEFAULT_H3_PRESETS, "light", "single", False)

    assert len(result["sigmas"].split(",")) == 17
