from .upscaler_loader import UpscaleModelLoaderFromHF
from .gguf_loader import GGUFModelLoaderFromHF
from .unet_loader import UNETModelLoaderFromHF
from .clip_loader import CLIPModelLoaderFromHF
from .vae_loader import VAEModelLoaderFromHF

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoaderFromHF": UpscaleModelLoaderFromHF,
    "GGUFModelLoaderFromHF": GGUFModelLoaderFromHF,
    "UNETModelLoaderFromHF": UNETModelLoaderFromHF,
    "CLIPModelLoaderFromHF": CLIPModelLoaderFromHF,
    "VAEModelLoaderFromHF": VAEModelLoaderFromHF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleModelLoaderFromHF": "Upscale Model Loader From HF",
    "GGUFModelLoaderFromHF": "GGUF Model Loader From HF",
    "UNETModelLoaderFromHF": "UNET Model Loader From HF",
    "CLIPModelLoaderFromHF": "CLIP Model Loader From HF",
    "VAEModelLoaderFromHF": "VAE Model Loader From HF"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
