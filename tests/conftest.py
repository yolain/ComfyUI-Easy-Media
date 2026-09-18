import sys
import types

if "folder_paths" not in sys.modules:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: "/tmp"
    folder_paths.get_output_directory = lambda: "/tmp"
    folder_paths.get_input_directory = lambda: "/tmp"
    folder_paths.get_annotated_filepath = lambda path: path
    folder_paths.models_dir = "/tmp/models"
    folder_paths.folder_names_and_paths = {}
    folder_paths.add_model_folder_path = lambda *_args: None
    folder_paths.get_save_image_path = lambda *args, **kwargs: ("/tmp", "test", 0, "", "")
    sys.modules["folder_paths"] = folder_paths

if "server" not in sys.modules:
    server = types.ModuleType("server")
    class _PromptServer:
        instance = None
    server.PromptServer = _PromptServer
    sys.modules["server"] = server
