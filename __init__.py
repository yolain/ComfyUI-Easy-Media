import os
import server
import yaml
import nodes as _nodes

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from aiohttp import web
from .nodes import *
from .routes import *

# Define the path
root_path = os.path.dirname(__file__)
project_name = os.path.basename(root_path)

web_default_version = 'release'
data = None
config_path = os.path.join(root_path, "config.yaml")
if os.path.isfile(config_path):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if data and "WEB_VERSION" in data:
        dist_path = f"dist/{data['WEB_VERSION']}"
    else:
        dist_path = f"dist/{web_default_version}"
else:
    dist_path = f"dist/{web_default_version}"

web_version = data.get("WEB_VERSION", web_default_version) if data else web_default_version
if not os.path.exists(os.path.join(root_path, dist_path)):
    print(f"web root {web_version} not found, using default")
    dist_path = f"dist/{web_default_version}"
    
dist_path = os.path.join(root_path, dist_path)
if os.path.exists(dist_path):
    # Add the routes for the extension
    server.PromptServer.instance.app.add_routes([
        web.static(f"/{project_name}/", dist_path),
    ])

    _nodes.EXTENSION_WEB_DIRS[project_name] = dist_path
else:
    print(f"Warning: Dist directory not found for {project_name}. Frontend assets will not be served.")

# Register extension
class EasyMediaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        nodes = [
            TimelineEditor,
            TimelineInfoOutput,
            TimelineSegmentOutput,
            TimelineSegmentCount,
            ImageIndexesToIntList,
            MakeImageList,
            MakeAudioList,
            # Wan
            BerniniModelPatch,
            # LTXV
            LTXVAddGuidesFromBatchIndexes,
            LTXVMakeRefVideo,
            # Video
            EasySaveVideo,
            EasyMergeVideos,
            EasyMergeVideosFromPaths,
        ]
        try:
            from comfy_extras.nodes_bernini import BerniniConditioning as CoreBerniniConditioning
        except ImportError:
            nodes.extend([BerniniConditioning])
        return nodes
async def comfy_entrypoint() -> EasyMediaExtension:
    return EasyMediaExtension()
