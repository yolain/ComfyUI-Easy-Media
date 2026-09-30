from __future__ import annotations

from typing import Any

import torch
from comfy_api.latest import io

from ..utils.h3_previous_frame import load_previous_frame
from ..utils.minimax import expand_image_inputs


class EasyH3LastFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="easy h3LastFrame", display_name="H3 Last Frame", category="EasyUse/H3/dev",
                         inputs=[io.Image.Input("images")], outputs=[io.Image.Output("image")], is_dev_only=True)

    @classmethod
    def execute(cls, images: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(images[-1:].detach().cpu().clone())



class EasyH3PreviousFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy h3PreviousFrame", display_name="H3 Previous Frame", category="EasyUse/H3/dev",
            is_input_list=True, is_dev_only=True, not_idempotent=True,
            inputs=[io.Image.Input("images"), io.String.Input("project_name"), io.Int.Input("segment_index"),
                    io.Int.Input("position"), io.Boolean.Input("resume", default=False),
                    io.AnyType.Input("previous", optional=True)],
            outputs=[io.Image.Output("images", is_output_list=True), io.String.Output("source")],
        )

    @classmethod
    def execute(cls, images: list, project_name: list[str], segment_index: list[int], position: list[int],
                resume: list[bool], previous: Any = None) -> io.NodeOutput:
        image, source = load_previous_frame(project_name[0], segment_index[0], resume[0])
        result = expand_image_inputs(images)
        if len(result) >= 9 or not 0 <= position[0] <= len(result):
            raise ValueError("Previous-frame reference exceeds the available image slots")
        result.insert(position[0], image)
        return io.NodeOutput(result, source)
