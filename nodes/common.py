from __future__ import annotations

from comfy_api.latest import io

CATEGORY_LOGIC = "EasyUse/Logic"
CATEGORY_LOADERS = "EasyUse/Loaders"
TYPE_FAST_MODEL_LOADER = io.Custom(io_type="FAST_MODEL_LOADER")


class EasyModelLoaderPack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy modelLoaderPack",
            display_name="Model Loader Pack",
            category=CATEGORY_LOADERS,
            description=(
                "Pack model components into a FAST_MODEL_LOADER compatible with "
                "FastUse nodes."
            ),
            inputs=[
                TYPE_FAST_MODEL_LOADER.Input("model_loader", optional=True),
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Vae.Input("audio_vae", optional=True),
            ],
            outputs=[TYPE_FAST_MODEL_LOADER.Output("model_loader")],
        )

    @classmethod
    def execute(
        cls,
        model_loader: dict | None = None,
        model: object | None = None,
        clip: object | None = None,
        vae: object | None = None,
        audio_vae: object | None = None,
    ) -> io.NodeOutput:
        if model_loader is not None:
            # Case 1: model_loader exists - use it as base, replace provided keys
            result = dict(model_loader)
            if model is not None:
                result["model"] = model
            if clip is not None:
                result["clip"] = clip
            if vae is not None:
                result["vae"] = vae
            if audio_vae is not None:
                result["audio_vae"] = audio_vae
            return io.NodeOutput(result)
        else:
            # Case 2: model_loader does not exist - model, clip, vae are required
            if model is None or clip is None or vae is None:
                raise ValueError("model, clip, and vae are required when model_loader is not provided")
            result = {
                "model": model,
                "clip": clip,
                "vae": vae,
            }
            if audio_vae is not None:
                result["audio_vae"] = audio_vae
            return io.NodeOutput(result)


class MatchLine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy matchLine",
            display_name="Match Line",
            category=CATEGORY_LOGIC,
            description="Return the zero-based index of the first line containing the match text.",
            inputs=[
                io.String.Input("text", default="", multiline=True),
                io.String.Input("match", default=""),
            ],
            outputs=[
                io.Int.Output("LINE_INDEX"),
            ],
        )

    @classmethod
    def execute(cls, text: str, match: str) -> io.NodeOutput:
        if not match:
            return io.NodeOutput(-1)

        line_index = next(
            (index for index, line in enumerate(text.splitlines()) if match in line),
            -1,
        )
        return io.NodeOutput(line_index)


class APIWorkflowGate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy apiWorkflowGate",
            display_name="API Workflow Gate",
            category=CATEGORY_LOGIC,
            description=(
                "Pass the input through for API workflow execution; return None without "
                "evaluating the input when execution includes workflow metadata."
            ),
            inputs=[
                io.AnyType.Input(
                    "value",
                    optional=True,
                    lazy=True,
                    tooltip="Any input to evaluate only when the execution prompt is API workflow format.",
                ),
            ],
            hidden=[io.Hidden.extra_pnginfo],
            outputs=[
                io.AnyType.Output("VALUE"),
                io.AnyType.Output("VALUES", is_output_list=True),
            ],
        )

    @classmethod
    def _is_workflow_format(cls) -> bool:
        hidden = getattr(cls, "hidden", None)
        extra_pnginfo = getattr(hidden, "extra_pnginfo", None)
        return _is_workflow_format(extra_pnginfo)

    @classmethod
    def check_lazy_status(cls, value: object | None = None) -> list[str]:
        if cls._is_workflow_format():
            return []
        if value is None:
            return ["value"]
        return []

    @classmethod
    def execute(cls, value: object | None = None) -> io.NodeOutput:
        if cls._is_workflow_format():
            return io.NodeOutput(None, [])
        if isinstance(value, list):
            return io.NodeOutput(None, value)
        return io.NodeOutput(value, [])


def _is_workflow_format(extra_pnginfo: object) -> bool:
    def _contains_workflow_metadata(value: object) -> bool:
        if isinstance(value, dict):
            if "workflow" in value and value["workflow"] is not None:
                return True
            return any(_contains_workflow_metadata(item) for item in value.values())
        if isinstance(value, list):
            return any(_contains_workflow_metadata(item) for item in value)
        return False

    return _contains_workflow_metadata(extra_pnginfo)
