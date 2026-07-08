from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ..utils import iter_valid_audio_inputs, merge_audio_inputs

CATEGORY_AUDIO = "EasyUse/Audio"
AUDIO_METHODS = ["add", "mean", "subtract", "multiply", "after", "before"]


def _first_list_value(value: Any, default: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return value


class EasyAudioMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy audioMerge",
            display_name="Merge Audio",
            category=CATEGORY_AUDIO,
            description="Merge or concatenate up to six audio inputs, expanding audio lists and ignoring empty list items.",
            is_input_list=True,
            inputs=[
                io.Autogrow.Input(
                    "audios",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"),
                        prefix="audio_",
                        min=1,
                        max=6,
                    ),
                    tooltip="Audio inputs to merge or concatenate. Connect up to six sources.",
                ),
                io.Combo.Input(
                    "merge_method",
                    options=AUDIO_METHODS,
                    default="add",
                    tooltip="Use add/mean/subtract/multiply to overlay audio, or after/before to concatenate audio.",
                    socketless=True,
                ),
            ],
            outputs=[
                io.Audio.Output("AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audios: io.Autogrow.Type,
        merge_method: str | list[str] = "add",
    ) -> io.NodeOutput:
        method = _first_list_value(merge_method, "add")
        audio_inputs = iter_valid_audio_inputs(audios)
        return io.NodeOutput(merge_audio_inputs(audio_inputs, str(method)))
