import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.prompt_builder import (
    I2V_TEMPLATE,
    MINIMAX_BASE_PROMPT,
    MINIMAX_REF_PROMPT,
    R2V_TEMPLATE,
    T2V_TEMPLATE,
    build_prompt_request,
    get_system_prompt_options,
)


def test_system_prompt_options_include_mode_and_image_rules():
    options = get_system_prompt_options()

    assert {
        "key": "default_t2v",
        "task_type": "t2v",
        "mode": "default",
        "min_images": 0,
        "max_images": 0,
        "system_prompt": T2V_TEMPLATE,
    } in options
    assert {
        "key": "default_i2v",
        "task_type": "i2v",
        "mode": "default",
        "min_images": 0,
        "max_images": None,
        "system_prompt": I2V_TEMPLATE,
    } in options


def test_system_prompt_options_include_ref_template_without_image_filtering():
    options = get_system_prompt_options()

    assert {
        "key": "ref_r2v",
        "task_type": "r2v",
        "mode": "ref",
        "min_images": 0,
        "max_images": None,
        "system_prompt": R2V_TEMPLATE,
    } in options


def test_system_prompt_options_include_minimax_mode_groups():
    options = get_system_prompt_options()

    assert {
        "key": "minimax_base",
        "format": "MiniMax",
        "modes": ["default", "l2v"],
        "system_prompt": MINIMAX_BASE_PROMPT,
    } in options
    assert {
        "key": "minimax_ref",
        "format": "MiniMax",
        "modes": ["ref", "edit"],
        "system_prompt": MINIMAX_REF_PROMPT,
    } in options


def test_custom_system_prompt_preserves_unknown_braced_text():
    custom_template = 'Write JSON like {"subject": "{character}"}. Prompt: {user_prompt}'

    _, prompt, _ = build_prompt_request(
        "v2v",
        "make it move",
        custom_system_prompt=custom_template,
    )

    assert prompt == 'Write JSON like {"subject": "{character}"}. Prompt: make it move'


def test_minimax_uses_mode_specific_system_prompts():
    for task_type in ("t2v", "i2v", "l2v"):
        system_prompt, user_prompt, json_mode = build_prompt_request(
            task_type, "make a video", video_format="MiniMax"
        )
        assert (system_prompt, user_prompt, json_mode) == (
            MINIMAX_BASE_PROMPT, "make a video", False
        )

    for task_type in ("r2v", "rv2v", "v2v", "vi2v"):
        system_prompt, user_prompt, json_mode = build_prompt_request(
            task_type, "change the video", video_format="MiniMax"
        )
        assert (system_prompt, user_prompt, json_mode) == (
            MINIMAX_REF_PROMPT, "change the video", False
        )

    system_prompt, _, _ = build_prompt_request(
        "custom-task",
        "make a video",
        video_format="MiniMax",
        task_mode="default",
    )
    assert system_prompt == MINIMAX_BASE_PROMPT
