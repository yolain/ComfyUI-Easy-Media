from __future__ import annotations

import os
from pathlib import Path

from comfy_api.latest import Types, io

from ..modules.subtitle_recognition import (
    SUBTITLE_RECOGNITION_METHODS,
    recognize_audio_subtitles,
)
from ..utils import (
    extract_video_audio_to_temp,
    save_audio_to_temp_wav,
    subtitle_segments_to_srt,
    subtitle_segments_to_timestamp_text,
    video_input_to_local_file,
)

CATEGORY_MEDIA = "EasyUse/Media"


class RecognizeSubtitle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy recognizeSubtitle",
            display_name="Recognize Subtitle",
            category=CATEGORY_MEDIA,
            description=(
                "Recognize timestamped subtitles from an AUDIO or VIDEO input "
                "using Qwen3-ASR or Whisper Large V3, and output normalized SRT text."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Optional AUDIO input. Takes priority when both inputs are connected.",
                ),
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip="Optional VIDEO input. Its audio track is extracted when AUDIO is not connected.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=SUBTITLE_RECOGNITION_METHODS,
                    default="whisper-large-v3",
                    tooltip="Choose the same ASR model used by multitrack subtitle recognition.",
                ),
                io.Combo.Input(
                    "output_format",
                    options=["srt", "timestamp"],
                    default="srt",
                    tooltip="Output standard SRT or one '(start, end) text' entry per line.",
                ),
                io.Int.Input(
                    "max_sentence_length",
                    default=20,
                    min=1,
                    max=500,
                    step=1,
                    tooltip="Maximum number of characters in each subtitle entry.",
                ),
                io.Boolean.Input(
                    "unload_model",
                    default=True,
                    tooltip="Move the ASR model to CPU and clear accelerator caches after recognition.",
                ),
            ],
            outputs=[io.String.Output("SUBTITLE_TEXT")],
        )

    @classmethod
    def execute(
        cls,
        audio: dict | None = None,
        video: object | None = None,
        model_type: str = "whisper-large-v3",
        output_format: str = "srt",
        max_sentence_length: int = 20,
        unload_model: bool = True,
    ) -> io.NodeOutput:
        temporary_paths: list[str | Path] = []
        try:
            if audio is not None:
                audio_path = save_audio_to_temp_wav(audio)
                if audio_path is None:
                    raise ValueError("Recognize Subtitle could not serialize the AUDIO input.")
                temporary_paths.append(audio_path)
            elif video is not None:
                video_path, video_temp_files = video_input_to_local_file(
                    video,
                    suffix=".mp4",
                    save_kwargs={
                        "format": Types.VideoContainer.AUTO,
                        "codec": Types.VideoCodec.AUTO,
                    },
                )
                temporary_paths.extend(video_temp_files)
                audio_path = extract_video_audio_to_temp(Path(video_path))
                temporary_paths.append(audio_path)
            else:
                raise ValueError("Recognize Subtitle requires an AUDIO or VIDEO input.")

            segments = recognize_audio_subtitles(
                Path(audio_path),
                str(model_type),
                int(max_sentence_length),
                bool(unload_model),
            )
            if output_format == "timestamp":
                subtitle_text = subtitle_segments_to_timestamp_text(segments)
            elif output_format == "srt":
                subtitle_text = subtitle_segments_to_srt(segments)
            else:
                raise ValueError("output_format must be srt or timestamp")
            return io.NodeOutput(subtitle_text)
        finally:
            for path in temporary_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
