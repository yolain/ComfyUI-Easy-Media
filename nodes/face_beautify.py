from __future__ import annotations

import hashlib
import json

import torch
from comfy_api.latest import io

from ..utils.face_beautify import (
    PARAMETER_LIMITS, beautify_image_batch, beauty_preview, beauty_source_path,
    resolve_numeric_beauty_preset,
    load_beauty_source, parse_beauty_preset,
)


class EasyFaceBeautify(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy faceBeautify", display_name="Face Beauty", category="EasyUse/Image",
            description="Interactive face reshaping and skin retouching. Use the connected IMAGE or select a portrait in the editor. Detects one face per image; no face leaves the image unchanged.",
            inputs=[io.Custom(io_type="EASY_FACE_BEAUTY").Input("preset"), io.Image.Input("image", optional=True)],
            outputs=[io.Image.Output("image"), io.String.Output("preset_json")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, preset: str = "{}", image: torch.Tensor | None = None) -> str:
        settings = parse_beauty_preset(preset)
        if (image is not None and settings.get("source_mode") != "picker") or not settings["source"]:
            return "connected-image"
        # Unresolved linked inputs can arrive as None during cache fingerprinting.
        # A missing picker file must not block an IMAGE connection overriding it.
        try:
            path = beauty_source_path(settings["source"])
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except (OSError, ValueError) as error:
            return f"unavailable-picker-source:{error}"

    @classmethod
    def execute(cls, preset: str = "{}", image: torch.Tensor | None = None) -> io.NodeOutput:
        settings = parse_beauty_preset(preset)
        if settings.get("source_mode") == "picker":
            image = None
        input_connected = image is not None
        if image is None:
            if not settings["source"]:
                raise ValueError("Select a portrait in Face Beauty or connect an IMAGE input.")
            image = load_beauty_source(settings["source"])
        result, points = beautify_image_batch(image, settings, collect_preview=True)
        preview = beauty_preview(image[0], points)
        preview["source"] = settings["source"]
        preview["input_connected"] = input_connected
        return io.NodeOutput(result, json.dumps(settings, ensure_ascii=False),
                             ui={"face_beauty": [preview]})


class EasyFaceBeautifyStatic(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy faceBeautifyStatic", display_name="Face Beauty (Static)", category="EasyUse/Image",
            description=("Face beauty with native numeric widgets. A connected JSON preset overrides only its specified "
                         "parameters and colors. The IMAGE input is always used. Connect the result to a preview or save node."),
            inputs=[
                io.Image.Input("image"),
                *[io.Float.Input(name, default=0.0, min=low, max=high, step=0.01,
                                 display_mode=io.NumberDisplay.number)
                  for name, (low, high) in PARAMETER_LIMITS.items()],
                io.String.Input("lip_color", default="#c4485f", tooltip="Lip color in #RRGGBB hexadecimal format."),
                io.String.Input("blush_color", default="#ff9f80", tooltip="Blush color in #RRGGBB hexadecimal format."),
                io.String.Input("preset", optional=True, force_input=True,
                                tooltip="Optional preset JSON from Face Beauty. Only included parameters and colors override the numeric controls."),
            ],
            outputs=[io.Image.Output("image"), io.String.Output("preset_json")],
            is_output_node=False,
        )

    @classmethod
    def execute(
        cls, image: torch.Tensor,
        slim_face: float = 0.0, chin: float = 0.0, eyes: float = 0.0,
        nose: float = 0.0, mouth: float = 0.0, eye_bright: float = 0.0,
        under_eye: float = 0.0, lip_tint: float = 0.0, blush: float = 0.0,
        smooth: float = 0.0, whiten: float = 0.0, brightness: float = 0.0, warmth: float = 0.0,
        lip_color: str = "#c4485f", blush_color: str = "#ff9f80",
        preset: str | None = None,
    ) -> io.NodeOutput:
        parameters = {
            "slim_face": slim_face, "chin": chin, "eyes": eyes, "nose": nose, "mouth": mouth,
            "eye_bright": eye_bright, "under_eye": under_eye, "lip_tint": lip_tint, "blush": blush,
            "smooth": smooth, "whiten": whiten, "brightness": brightness, "warmth": warmth,
        }
        colors = {"lip": lip_color, "blush": blush_color}
        settings = resolve_numeric_beauty_preset(parameters, colors, preset)
        result, _points = beautify_image_batch(image, settings)
        return io.NodeOutput(result, json.dumps(settings, ensure_ascii=False))
