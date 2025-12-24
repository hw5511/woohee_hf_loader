import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Union
from collections.abc import Iterable
import comfy.utils
import comfy.sd


class Folders:
    HF_VAE_CACHE_DIR = "hf_vae_cache_dir"


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


folder_paths.add_model_folder_path(Folders.HF_VAE_CACHE_DIR, str(Path(folder_paths.models_dir) / Folders.HF_VAE_CACHE_DIR))
add_extension_to_folder_path(Folders.HF_VAE_CACHE_DIR, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.HF_VAE_CACHE_DIR))


class VAEModelLoaderFromHF:
    """
    ComfyUI Custom Node for loading VAE Models from Hugging Face Hub

    This node allows you to load VAE models directly from Hugging Face
    without manually downloading them to your models folder.
    Optimized for Qwen Image Layered models.
    """

    def __init__(self):
        self.loaded_vae_model = None

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
                    "default": "qwen_image_layered_vae.safetensors",
                    "multiline": False
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "split_files/vae",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae_from_hf"
    CATEGORY = "HF_loaders"

    def load_vae_from_hf(self, repo_name, filename, subfolder="split_files/vae"):
        """
        Load VAE model from Hugging Face Hub.

        Args:
            repo_name (str): Hugging Face repository name
            filename (str): Model file name in the repository
            subfolder (str, optional): Subfolder path within the repository

        Returns:
            tuple: Loaded VAE model
        """
        subfolder_path = subfolder.strip() if subfolder else None

        cache_key = (repo_name, filename, subfolder_path)

        if self.loaded_vae_model is not None:
            if self.loaded_vae_model[0] == cache_key:
                print(f"Using cached VAE model from {repo_name}/{subfolder_path + '/' if subfolder_path else ''}{filename}")
                return (self.loaded_vae_model[1],)
            else:
                temp = self.loaded_vae_model
                self.loaded_vae_model = None
                del temp

        cache_dirs = folder_paths.get_folder_paths(Folders.HF_VAE_CACHE_DIR)

        download_params = {
            "repo_id": repo_name,
            "filename": filename,
            "cache_dir": cache_dirs[0]
        }

        if subfolder_path:
            download_params["subfolder"] = subfolder_path

        model_path = hf_hub_download(**download_params)

        print(f"Loaded VAE model from {model_path}")

        # Load VAE state dict
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        # Create VAE object using ComfyUI's SD module
        vae_model = comfy.sd.VAE(sd=sd)

        self.loaded_vae_model = (cache_key, vae_model)

        return (vae_model,)
