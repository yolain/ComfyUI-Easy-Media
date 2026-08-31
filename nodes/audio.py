from __future__ import annotations

from typing import Any

from comfy_api.latest import io

from ..utils import iter_valid_audio_inputs, merge_audio_inputs, silence, split_list_outputs

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


class MakeAudioList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy makeAudioList",
            display_name="Make Audio List",
            category=CATEGORY_AUDIO,
            description="Combine up to 10 optional audio inputs into an audio list.",
            inputs=[
                io.Boolean.Input("skip_empty", default=False, label_on="Skip", label_off="Fill"),
                io.Audio.Input("audio1", optional=True),
                io.Audio.Input("audio2", optional=True),
                io.Audio.Input("audio3", optional=True),
                io.Audio.Input("audio4", optional=True),
                io.Audio.Input("audio5", optional=True),
                io.Audio.Input("audio6", optional=True),
                io.Audio.Input("audio7", optional=True),
                io.Audio.Input("audio8", optional=True),
                io.Audio.Input("audio9", optional=True),
                io.Audio.Input("audio10", optional=True),
            ],
            outputs=[
                io.Audio.Output("AUDIO", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, skip_empty: bool, **kwargs: object) -> io.NodeOutput:
        audios: list[dict] = []
        for i in range(1, 11):
            key = f"audio{i}"
            v = kwargs.get(key)
            if v is not None:
                audios.append(v)
            elif not skip_empty:
                empty = silence(16000, 0.001, 1)
                audios.append({"waveform": empty, "sample_rate": 16000})

        return io.NodeOutput(audios)

class SplitAudios(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy splitAudios",
            display_name="Split Audios",
            category=CATEGORY_AUDIO,
            description="Split an audio list into 10 single-audio outputs.",
            is_input_list=True,
            inputs=[
                io.Audio.Input("audios"),
            ],
            outputs=[
                io.Audio.Output(f"AUDIO{i}") for i in range(0, 10)
            ],
        )

    @classmethod
    def execute(cls, audios: list[dict[str, object]]) -> io.NodeOutput:
        if not audios:
            raise ValueError("audios must contain at least one audio.")
        if not all(isinstance(audio, dict) for audio in audios):
            raise TypeError("audios must contain only audio dictionaries.")

        return io.NodeOutput(*split_list_outputs(audios))
