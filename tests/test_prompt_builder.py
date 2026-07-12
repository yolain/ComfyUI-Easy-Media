import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.prompt_builder import (
    I2V_TEMPLATE,
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


def test_custom_system_prompt_preserves_unknown_braced_text():
    custom_template = 'Write JSON like {"subject": "{character}"}. Prompt: {user_prompt}'

    _, prompt, _ = build_prompt_request(
        "v2v",
        "make it move",
        custom_system_prompt=custom_template,
    )

    assert prompt == 'Write JSON like {"subject": "{character}"}. Prompt: make it move'
