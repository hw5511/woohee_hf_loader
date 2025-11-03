import comfy.utils
import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Union
from collections.abc import Iterable
import torch


class Folders:
    HF_CACHE_DIR = "hf_cache_dir"


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


# Setup HF cache directory
folder_paths.add_model_folder_path(Folders.HF_CACHE_DIR, str(Path(folder_paths.models_dir) / Folders.HF_CACHE_DIR))
add_extension_to_folder_path(Folders.HF_CACHE_DIR, folder_paths.supported_pt_extensions)
try_mkdir(str(Path(folder_paths.models_dir) / Folders.HF_CACHE_DIR))


class UpscaleModelLoaderFromHF:
    """
    ComfyUI Custom Node for loading Upscale Models from Hugging Face Hub

    This node allows you to load upscale models directly from Hugging Face
    without manually downloading them to your models folder.
    """

    def __init__(self):
        self.loaded_upscale_model = None

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

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_upscale_model_from_hf"
    CATEGORY = "HF_loaders"

    def load_upscale_model_from_hf(self, repo_name, filename, subfolder=""):
        """
        Load upscale model from Hugging Face Hub.

        Args:
            repo_name (str): Hugging Face repository name
            filename (str): Model file name in the repository
            subfolder (str, optional): Subfolder path within the repository

        Returns:
            tuple: Loaded upscale model
        """
        # Normalize subfolder (remove if empty)
        subfolder_path = subfolder.strip() if subfolder else None

        # Create cache key including subfolder
        cache_key = (repo_name, filename, subfolder_path)

        # Use cached model if available
        if self.loaded_upscale_model is not None:
            if self.loaded_upscale_model[0] == cache_key:
                print(f"Using cached upscale model from {repo_name}/{subfolder_path + '/' if subfolder_path else ''}{filename}")
                return (self.loaded_upscale_model[1],)
            else:
                # Clear old cache
                temp = self.loaded_upscale_model
                self.loaded_upscale_model = None
                del temp

        # Download from Hugging Face (public repositories only)
        cache_dirs = folder_paths.get_folder_paths(Folders.HF_CACHE_DIR)

        # Build download parameters
        download_params = {
            "repo_id": repo_name,
            "filename": filename,
            "cache_dir": cache_dirs[0]
        }

        # Add subfolder if specified
        if subfolder_path:
            download_params["subfolder"] = subfolder_path

        model_path = hf_hub_download(**download_params)

        print(f"Loaded upscale model from {model_path}")

        # Load the model using ComfyUI's utility function
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        # Handle model architecture variations
        # Some models have 'module.' prefix in state_dict keys
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd.keys():
            sd = {k.replace('module.', ''): v for k, v in sd.items()}

        # Initialize the upscale model
        # ComfyUI uses different architectures for upscale models
        # We'll detect the architecture and load accordingly
        upscale_model = self._initialize_model(sd)

        # Cache the loaded model with cache key
        self.loaded_upscale_model = (cache_key, upscale_model)

        return (upscale_model,)

    def _initialize_model(self, state_dict):
        """
        Initialize upscale model from state dict.

        Args:
            state_dict: Model state dictionary

        Returns:
            Initialized model
        """
        # Try to import ComfyUI's upscale model classes
        try:
            from comfy_extras.chainner_models import model_loading

            # Detect and load the appropriate model architecture
            model = model_loading.load_state_dict(state_dict)

            if model is None:
                raise Exception("Could not detect model architecture")

            model.eval()
            return model

        except ImportError:
            # Fallback: create a simple wrapper if ComfyUI's extras aren't available
            print("Warning: comfy_extras not found, using fallback loader")

            class UpscaleModelWrapper:
                def __init__(self, sd):
                    self.state_dict = sd

            return UpscaleModelWrapper(state_dict)
