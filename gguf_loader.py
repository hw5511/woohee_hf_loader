import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Union
from collections.abc import Iterable


class Folders:
    HF_GGUF_CACHE_DIR = "hf_gguf_cache_dir"


def add_extension_to_folder_path(folder_name: str, extensions: Union[str, list[str]]):
    if folder_name in folder_paths.folder_names_and_paths:
        if isinstance(extensions, str):
            folder_paths.folder_names_and_paths[folder_name][1].add(extensions)
        elif isinstance(extensions, Iterable):
            for ext in extensions:
                folder_paths.folder_names_and_paths[folder_name][1].add(ext)


def try_mkdir(full_path: str):
    try:
        Path(full_path).mkdir()
    except Exception:
        pass


folder_paths.add_model_folder_path(Folders.HF_GGUF_CACHE_DIR, str(Path(folder_paths.models_dir) / Folders.HF_GGUF_CACHE_DIR))
add_extension_to_folder_path(Folders.HF_GGUF_CACHE_DIR, ".gguf")
try_mkdir(str(Path(folder_paths.models_dir) / Folders.HF_GGUF_CACHE_DIR))


class GGUFModelLoaderFromHF:
    def __init__(self):
        self.loaded_gguf_model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_name": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "filename": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_gguf_model_from_hf"
    CATEGORY = "HF_loaders"

    def load_gguf_model_from_hf(self, repo_name, filename, subfolder=""):
        subfolder_path = subfolder.strip() if subfolder else None

        cache_key = (repo_name, filename, subfolder_path)

        if self.loaded_gguf_model is not None:
            if self.loaded_gguf_model[0] == cache_key:
                print(f"Using cached GGUF model from {repo_name}/{subfolder_path + '/' if subfolder_path else ''}{filename}")
                return (self.loaded_gguf_model[1],)
            else:
                temp = self.loaded_gguf_model
                self.loaded_gguf_model = None
                del temp

        cache_dirs = folder_paths.get_folder_paths(Folders.HF_GGUF_CACHE_DIR)

        download_params = {
            "repo_id": repo_name,
            "filename": filename,
            "cache_dir": cache_dirs[0]
        }

        if subfolder_path:
            download_params["subfolder"] = subfolder_path

        model_path = hf_hub_download(**download_params)

        print(f"Loaded GGUF model from {model_path}")

        try:
            from comfy.model_management import unet_manual_cast
            from nodes import UNETLoader

            unet_loader = UNETLoader()
            model_tuple = unet_loader.load_unet(model_path)
            gguf_model = model_tuple[0]

        except Exception as e:
            print(f"Warning: Could not load GGUF model with ComfyUI's UNETLoader: {e}")

            class GGUFModelWrapper:
                def __init__(self, path):
                    self.path = path

            gguf_model = GGUFModelWrapper(model_path)

        self.loaded_gguf_model = (cache_key, gguf_model)

        return (gguf_model,)
