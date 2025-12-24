import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Union
from collections.abc import Iterable
import comfy.sd


class Folders:
    HF_CLIP_CACHE_DIR = "hf_clip_cache_dir"


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


folder_paths.add_model_folder_path(Folders.HF_CLIP_CACHE_DIR, str(Path(folder_paths.models_dir) / Folders.HF_CLIP_CACHE_DIR))
add_extension_to_folder_path(Folders.HF_CLIP_CACHE_DIR, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.HF_CLIP_CACHE_DIR))


class CLIPModelLoaderFromHF:
    """
    ComfyUI Custom Node for loading CLIP Models from Hugging Face Hub

    This node allows you to load CLIP models directly from Hugging Face
    without manually downloading them to your models folder.
    Optimized for Qwen 2.5 VL models.
    """

    def __init__(self):
        self.loaded_clip_model = None

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
                    "default": "Comfy-Org/Qwen-Image_ComfyUI",
                    "multiline": False
                }),
                "filename": ("STRING", {
                    "default": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                    "multiline": False
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "split_files/text_encoders",
                    "multiline": False
                }),
                "type": ("STRING", {
                    "default": "qwen_image",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_from_hf"
    CATEGORY = "HF_loaders"

    def load_clip_from_hf(self, repo_name, filename, subfolder="split_files/text_encoders", type="qwen_image"):
        """
        Load CLIP model from Hugging Face Hub.

        Args:
            repo_name (str): Hugging Face repository name
            filename (str): Model file name in the repository
            subfolder (str, optional): Subfolder path within the repository
            type (str, optional): CLIP type for ComfyUI

        Returns:
            tuple: Loaded CLIP model
        """
        subfolder_path = subfolder.strip() if subfolder else None

        cache_key = (repo_name, filename, subfolder_path, type)

        if self.loaded_clip_model is not None:
            if self.loaded_clip_model[0] == cache_key:
                print(f"Using cached CLIP model from {repo_name}/{subfolder_path + '/' if subfolder_path else ''}{filename}")
                return (self.loaded_clip_model[1],)
            else:
                temp = self.loaded_clip_model
                self.loaded_clip_model = None
                del temp

        cache_dirs = folder_paths.get_folder_paths(Folders.HF_CLIP_CACHE_DIR)

        download_params = {
            "repo_id": repo_name,
            "filename": filename,
            "cache_dir": cache_dirs[0]
        }

        if subfolder_path:
            download_params["subfolder"] = subfolder_path

        model_path = hf_hub_download(**download_params)

        print(f"Loaded CLIP model from {model_path}")

        # Load CLIP using ComfyUI's SD module
        clip_model = comfy.sd.load_clip(
            ckpt_paths=[model_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=type
        )

        self.loaded_clip_model = (cache_key, clip_model)

        return (clip_model,)
