import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Union
from collections.abc import Iterable


class Folders:
    HF_UNET_CACHE_DIR = "hf_unet_cache_dir"


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


folder_paths.add_model_folder_path(Folders.HF_UNET_CACHE_DIR, str(Path(folder_paths.models_dir) / Folders.HF_UNET_CACHE_DIR))
add_extension_to_folder_path(Folders.HF_UNET_CACHE_DIR, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.HF_UNET_CACHE_DIR))


class UNETModelLoaderFromHF:
    """
    ComfyUI Custom Node for loading UNET Models from Hugging Face Hub

    This node allows you to load UNET models directly from Hugging Face
    without manually downloading them to your models folder.
    Optimized for Qwen Image Layered models.
    """

    def __init__(self):
        self.loaded_unet_model = None

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input types for the node.

        Returns:
            dict: Dictionary of input specifications
        """
        return {
            "required": {
                "repo_name": ("STRING", {
                    "default": "Comfy-Org/Qwen-Image-Layered_ComfyUI",
                    "multiline": False
                }),
                "filename": ("STRING", {
                    "default": "qwen_image_layered_fp8mixed.safetensors",
                    "multiline": False
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "split_files/diffusion_models",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet_from_hf"
    CATEGORY = "HF_loaders"

    def load_unet_from_hf(self, repo_name, filename, subfolder="split_files/diffusion_models"):
        """
        Load UNET model from Hugging Face Hub.

        Args:
            repo_name (str): Hugging Face repository name
            filename (str): Model file name in the repository
            subfolder (str, optional): Subfolder path within the repository

        Returns:
            tuple: Loaded UNET model
        """
        subfolder_path = subfolder.strip() if subfolder else None

        cache_key = (repo_name, filename, subfolder_path)

        if self.loaded_unet_model is not None:
            if self.loaded_unet_model[0] == cache_key:
                print(f"Using cached UNET model from {repo_name}/{subfolder_path + '/' if subfolder_path else ''}{filename}")
                return (self.loaded_unet_model[1],)
            else:
                temp = self.loaded_unet_model
                self.loaded_unet_model = None
                del temp

        cache_dirs = folder_paths.get_folder_paths(Folders.HF_UNET_CACHE_DIR)

        download_params = {
            "repo_id": repo_name,
            "filename": filename,
            "cache_dir": cache_dirs[0]
        }

        if subfolder_path:
            download_params["subfolder"] = subfolder_path

        model_path = hf_hub_download(**download_params)

        print(f"Loaded UNET model from {model_path}")

        try:
            from nodes import UNETLoader

            unet_loader = UNETLoader()
            model_tuple = unet_loader.load_unet(model_path)
            unet_model = model_tuple[0]

        except Exception as e:
            print(f"Warning: Could not load UNET model with ComfyUI's UNETLoader: {e}")

            class UNETModelWrapper:
                def __init__(self, path):
                    self.path = path

            unet_model = UNETModelWrapper(model_path)

        self.loaded_unet_model = (cache_key, unet_model)

        return (unet_model,)
