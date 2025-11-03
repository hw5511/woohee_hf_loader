from .upscaler_loader import UpscaleModelLoaderFromHF

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoaderFromHF": UpscaleModelLoaderFromHF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscaleModelLoaderFromHF": "Upscale Model Loader From HF"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
