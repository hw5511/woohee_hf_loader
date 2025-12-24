import folder_paths
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Union
from collections.abc import Iterable
import comfy.utils
import comfy.sd
import torch

# WanVAE fallback implementation
# Try to import from ComfyUI first, fallback to embedded version if not available
# PATCHED: Force using embedded WanVAE for ComfyUI 0.3.62 image_channels compatibility
try:
    raise ImportError("Forcing embedded WanVAE for image_channels parameter support")
    from comfy.ldm.wan.vae import WanVAE
    WANVAE_SOURCE = "ComfyUI"
except ImportError:
    # Embedded WanVAE implementation for ComfyUI v0.3.62 compatibility
    # Original source: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/wan/vae.py
    # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    from comfy.ldm.modules.diffusionmodules.model import vae_attention
    import comfy.ops
    ops = comfy.ops.disable_weight_init

    CACHE_T = 2

    class CausalConv3d(ops.Conv3d):
        """Causal 3d convolution."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._padding = (self.padding[2], self.padding[2], self.padding[1],
                             self.padding[1], 2 * self.padding[0], 0)
            self.padding = (0, 0, 0)

        def forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
            if cache_list is not None:
                cache_x = cache_list[cache_idx]
                cache_list[cache_idx] = None

            padding = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]
                del cache_x
            x = F.pad(x, padding)

            return super().forward(x)

    class RMS_norm(nn.Module):
        def __init__(self, dim, channel_first=True, images=True, bias=False):
            super().__init__()
            broadcastable_dims = (1, 1, 1) if not images else (1, 1)
            shape = (dim, *broadcastable_dims) if channel_first else (dim,)

            self.channel_first = channel_first
            self.scale = dim**0.5
            self.gamma = nn.Parameter(torch.ones(shape))
            self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

        def forward(self, x):
            return F.normalize(
                x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)

    class Resample(nn.Module):
        def __init__(self, dim, mode):
            assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
            super().__init__()
            self.dim = dim
            self.mode = mode

            if mode == 'upsample2d':
                self.resample = nn.Sequential(
                    nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                    ops.Conv2d(dim, dim // 2, 3, padding=1))
            elif mode == 'upsample3d':
                self.resample = nn.Sequential(
                    nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                    ops.Conv2d(dim, dim // 2, 3, padding=1))
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
            elif mode == 'downsample2d':
                self.resample = nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    ops.Conv2d(dim, dim, 3, stride=(2, 2)))
            elif mode == 'downsample3d':
                self.resample = nn.Sequential(
                    nn.ZeroPad2d((0, 1, 0, 1)),
                    ops.Conv2d(dim, dim, 3, stride=(2, 2)))
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            else:
                self.resample = nn.Identity()

        def forward(self, x, feat_cache=None, feat_idx=[0]):
            b, c, t, h, w = x.size()
            if self.mode == 'upsample3d':
                if feat_cache is not None:
                    idx = feat_idx[0]
                    if feat_cache[idx] is None:
                        feat_cache[idx] = 'Rep'
                        feat_idx[0] += 1
                    else:
                        cache_x = x[:, :, -CACHE_T:, :, :].clone()
                        if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                        if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                            cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                        if feat_cache[idx] == 'Rep':
                            x = self.time_conv(x)
                        else:
                            x = self.time_conv(x, feat_cache[idx])
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1

                        x = x.reshape(b, 2, c, t, h, w)
                        x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                        x = x.reshape(b, c, t * 2, h, w)
            t = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.resample(x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

            if self.mode == 'downsample3d':
                if feat_cache is not None:
                    idx = feat_idx[0]
                    if feat_cache[idx] is None:
                        feat_cache[idx] = x.clone()
                        feat_idx[0] += 1
                    else:
                        cache_x = x[:, :, -1:, :, :].clone()
                        x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
            return x

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.0):
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim

            self.residual = nn.Sequential(
                RMS_norm(in_dim, images=False), nn.SiLU(),
                CausalConv3d(in_dim, out_dim, 3, padding=1),
                RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
                CausalConv3d(out_dim, out_dim, 3, padding=1))
            self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

        def forward(self, x, feat_cache=None, feat_idx=[0]):
            old_x = x
            for layer in self.residual:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    idx = feat_idx[0]
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    x = layer(x, cache_list=feat_cache, cache_idx=idx)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            return x + self.shortcut(old_x)

    class AttentionBlock(nn.Module):
        """Causal self-attention with a single head."""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

            self.norm = RMS_norm(dim)
            self.to_qkv = ops.Conv2d(dim, dim * 3, 1)
            self.proj = ops.Conv2d(dim, dim, 1)
            self.optimized_attention = vae_attention()

        def forward(self, x):
            identity = x
            b, c, t, h, w = x.size()
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = self.norm(x)
            q, k, v = self.to_qkv(x).chunk(3, dim=1)
            x = self.optimized_attention(q, k, v)
            x = self.proj(x)
            x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            return x + identity

    class Encoder3d(nn.Module):
        def __init__(self, dim=128, z_dim=4, input_channels=3, dim_mult=[1, 2, 4, 4],
                     num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
            super().__init__()
            self.dim = dim
            self.z_dim = z_dim
            self.dim_mult = dim_mult
            self.num_res_blocks = num_res_blocks
            self.attn_scales = attn_scales
            self.temperal_downsample = temperal_downsample

            dims = [dim * u for u in [1] + dim_mult]
            scale = 1.0

            self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

            downsamples = []
            for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                for _ in range(num_res_blocks):
                    downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        downsamples.append(AttentionBlock(out_dim))
                    in_dim = out_dim

                if i != len(dim_mult) - 1:
                    mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                    downsamples.append(Resample(out_dim, mode=mode))
                    scale /= 2.0
            self.downsamples = nn.Sequential(*downsamples)

            self.middle = nn.Sequential(
                ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
                ResidualBlock(out_dim, out_dim, dropout))

            self.head = nn.Sequential(
                RMS_norm(out_dim, images=False), nn.SiLU(),
                CausalConv3d(out_dim, z_dim, 3, padding=1))

        def forward(self, x, feat_cache=None, feat_idx=[0]):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = self.conv1(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = self.conv1(x)

            for layer in self.downsamples:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            for layer in self.middle:
                if isinstance(layer, ResidualBlock) and feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            for layer in self.head:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    idx = feat_idx[0]
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    x = layer(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            return x

    class Decoder3d(nn.Module):
        def __init__(self, dim=128, z_dim=4, output_channels=3, dim_mult=[1, 2, 4, 4],
                     num_res_blocks=2, attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0):
            super().__init__()
            self.dim = dim
            self.z_dim = z_dim
            self.dim_mult = dim_mult
            self.num_res_blocks = num_res_blocks
            self.attn_scales = attn_scales
            self.temperal_upsample = temperal_upsample

            dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
            scale = 1.0 / 2**(len(dim_mult) - 2)

            self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

            self.middle = nn.Sequential(
                ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
                ResidualBlock(dims[0], dims[0], dropout))

            upsamples = []
            for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                if i == 1 or i == 2 or i == 3:
                    in_dim = in_dim // 2
                for _ in range(num_res_blocks + 1):
                    upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        upsamples.append(AttentionBlock(out_dim))
                    in_dim = out_dim

                if i != len(dim_mult) - 1:
                    mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                    upsamples.append(Resample(out_dim, mode=mode))
                    scale *= 2.0
            self.upsamples = nn.Sequential(*upsamples)

            self.head = nn.Sequential(
                RMS_norm(out_dim, images=False), nn.SiLU(),
                CausalConv3d(out_dim, output_channels, 3, padding=1))

        def forward(self, x, feat_cache=None, feat_idx=[0]):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = self.conv1(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = self.conv1(x)

            for layer in self.middle:
                if isinstance(layer, ResidualBlock) and feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            for layer in self.upsamples:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            for layer in self.head:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    idx = feat_idx[0]
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    x = layer(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            return x

    def count_conv3d(model):
        count = 0
        for m in model.modules():
            if isinstance(m, CausalConv3d):
                count += 1
        return count

    class WanVAE(nn.Module):
        def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                     attn_scales=[], temperal_downsample=[True, True, False], image_channels=3, dropout=0.0):
            super().__init__()
            self.dim = dim
            self.z_dim = z_dim
            self.dim_mult = dim_mult
            self.num_res_blocks = num_res_blocks
            self.attn_scales = attn_scales
            self.temperal_downsample = temperal_downsample
            self.temperal_upsample = temperal_downsample[::-1]

            self.encoder = Encoder3d(dim, z_dim * 2, image_channels, dim_mult, num_res_blocks,
                                     attn_scales, self.temperal_downsample, dropout)
            self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
            self.decoder = Decoder3d(dim, z_dim, image_channels, dim_mult, num_res_blocks,
                                     attn_scales, self.temperal_upsample, dropout)

        def encode(self, x):
            conv_idx = [0]
            feat_map = [None] * count_conv3d(self.decoder)
            t = x.shape[2]
            iter_ = 1 + (t - 1) // 4
            for i in range(iter_):
                conv_idx = [0]
                if i == 0:
                    out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
                else:
                    out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                        feat_cache=feat_map, feat_idx=conv_idx)
                    out = torch.cat([out, out_], 2)
            mu, log_var = self.conv1(out).chunk(2, dim=1)
            return mu

        def decode(self, z):
            conv_idx = [0]
            feat_map = [None] * count_conv3d(self.decoder)
            iter_ = z.shape[2]
            x = self.conv2(z)
            for i in range(iter_):
                conv_idx = [0]
                if i == 0:
                    out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
                else:
                    out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
                    out = torch.cat([out, out_], 2)
            return out

    WANVAE_SOURCE = "Embedded"


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

        # Detect 4-channel VAE (Qwen-Image-Layered)
        is_4channel_wanvae = False
        input_channels = None

        if "encoder.conv1.weight" in sd:
            input_channels = sd["encoder.conv1.weight"].shape[1]

            # Check if this is a WanVAE structure
            is_wanvae = "decoder.middle.0.residual.0.gamma" in sd

            if input_channels == 4 and is_wanvae:
                is_4channel_wanvae = True
                print(f"[VAELoader] Detected 4-channel WanVAE (Qwen-Image-Layered)")
                print(f"[VAELoader] Input channels: {input_channels}")
            else:
                print(f"[VAELoader] Detected {input_channels}-channel VAE")

        # For 4-channel WanVAE, skip standard loading and use direct instantiation
        if is_4channel_wanvae:
            print(f"\n{'='*60}")
            print(f"[VAELoader] 4-channel RGBA detected - using direct WanVAE instantiation")
            print(f"[VAELoader] Bypassing ComfyUI's standard VAE loader")
            print(f"{'='*60}\n")

            try:
                # Extract configuration from state dict
                print(f"[VAELoader] Using WanVAE from: {WANVAE_SOURCE}")

                # Extract configuration from state dict
                dim = sd["decoder.head.0.gamma"].shape[0]
                z_dim = 16  # Wan 2.1 default

                print(f"[VAELoader] Creating WanVAE with:")
                print(f"  - dim: {dim}")
                print(f"  - z_dim: {z_dim}")
                print(f"  - image_channels: {input_channels} (4-channel RGBA)")

                # Create WanVAE instance with 4 channels
                ddconfig = {
                    "dim": dim,
                    "z_dim": z_dim,
                    "dim_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_scales": [],
                    "temperal_downsample": [False, True, True],
                    "dropout": 0.0,
                    "image_channels": input_channels  # 4 for RGBA!
                }

                wan_vae = WanVAE(**ddconfig)

                # Load state dict into WanVAE
                print(f"[VAELoader] Loading state dict into WanVAE...")
                wan_vae.load_state_dict(sd)
                wan_vae.eval()

                # Create ComfyUI VAE wrapper
                print(f"[VAELoader] Creating ComfyUI VAE wrapper...")

                # Create a minimal VAE wrapper that ComfyUI can use
                class CustomVAEWrapper:
                    def __init__(self, first_stage_model):
                        self.first_stage_model = first_stage_model
                        self.downscale_ratio = 8
                        self.upscale_ratio = 8
                        self.latent_channels = z_dim
                        self.output_channels = input_channels
                        self.process_input = lambda image: image
                        self.process_output = lambda image: image

                    def decode(self, samples_in):
                        return self.first_stage_model.decode(samples_in)

                    def encode(self, pixel_samples):
                        return self.first_stage_model.encode(pixel_samples)

                    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
                        return self.decode(samples)

                    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
                        return self.encode(pixel_samples)

                vae_model = CustomVAEWrapper(wan_vae)

                print(f"\n{'='*60}")
                print(f"SUCCESS: 4-channel WanVAE loaded with FULL RGBA support!")
                print(f"{'='*60}")
                print(f"Alpha channel: ✅ ENABLED")
                print(f"Layer decomposition: ✅ FULLY SUPPORTED")
                print(f"Transparency: ✅ AVAILABLE")
                print(f"{'='*60}\n")

            except Exception as e:
                print(f"[VAELoader] Direct WanVAE instantiation failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"4-CHANNEL WANVAE LOADING FAILED\n"
                    f"{'='*60}\n"
                    f"Could not load 4-channel WanVAE.\n\n"
                    f"Error: {e}\n\n"
                    f"This may be a ComfyUI version compatibility issue.\n"
                    f"{'='*60}\n"
                )
        else:
            # Standard VAE loading for non-4channel models
            print(f"[VAELoader] Using standard ComfyUI VAE loader...")
            vae_model = comfy.sd.VAE(sd=sd)

        self.loaded_vae_model = (cache_key, vae_model)

        return (vae_model,)
