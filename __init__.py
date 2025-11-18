from .upscaler_loader import UpscaleModelLoaderFromHF
from .gguf_loader import GGUFModelLoaderFromHF

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoaderFromHF": UpscaleModelLoaderFromHF,
    "GGUFModelLoaderFromHF": GGUFModelLoaderFromHF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleModelLoaderFromHF": "Upscale Model Loader From HF",
    "GGUFModelLoaderFromHF": "GGUF Model Loader From HF"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
