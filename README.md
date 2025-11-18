# Woohee HF Loader

Load models directly from Hugging Face Hub in ComfyUI without manual downloads.

## Description

This custom node collection allows you to load various models (upscalers, checkpoints, LoRAs, etc.) directly from Hugging Face repositories into ComfyUI. No need to manually download and place models in your models folder - just provide the repository name and filename!

Currently supports:
- Upscale models (ESRGAN, Real-ESRGAN, etc.)
- GGUF models (Quantized models for efficient inference)

## Features

- Load upscale models and GGUF models directly from Hugging Face Hub
- Automatic model caching to avoid repeated downloads
- Support for both public and private repositories (with API token)
- Compatible with standard ComfyUI model formats
- Works with ImageUpscaleWithModel node and other ComfyUI nodes

## Installation

### Via ComfyUI Manager

Search for "Woohee HF Loader" in ComfyUI Manager and install.

### Manual Installation

1. Navigate to your ComfyUI custom_nodes directory
2. Clone this repository:
   ```bash
   git clone https://github.com/hw5511/woohee_hf_loader.git
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Restart ComfyUI

## Usage

### Upscale Model Loader From HF

1. Add "Upscale Model Loader From HF" node to your workflow
2. Enter the Hugging Face repository name (e.g., `Phips/2xNomosUni_span_multijpg`)
3. Enter the model filename (e.g., `2xNomosUni_span_multijpg.pth`)
4. (Optional) Enter subfolder path if the model is in a subdirectory
5. Connect the output to an "Upscale Image (using Model)" node

### GGUF Model Loader From HF

1. Add "GGUF Model Loader From HF" node to your workflow
2. Enter the Hugging Face repository name (e.g., `QuantStack/Qwen-Image-Edit-2509-GGUF`)
3. Enter the GGUF filename (e.g., `Qwen-Image-Edit-2509-Q4_K_M.gguf`)
4. (Optional) Enter subfolder path if needed
5. Connect the MODEL output to compatible ComfyUI nodes

## Inputs

All loaders share the same input structure:
- **repo_name** (STRING): Hugging Face repository name
- **filename** (STRING): Model file name in the repository
- **subfolder** (STRING, optional): Subfolder path within the repository

## Outputs

- **Upscale Model Loader**: UPSCALE_MODEL output
- **GGUF Model Loader**: MODEL output

## Example Workflows

### Upscale Workflow
```
Load Image -> Upscale Model Loader From HF -> Upscale Image (using Model) -> Save Image
```

### GGUF Workflow
```
GGUF Model Loader From HF -> [Connect to compatible MODEL input nodes]
```

## Supported Model Formats

- **Upscale Models**: PyTorch models (.pth, .pt), SafeTensors (.safetensors)
- **GGUF Models**: Quantized GGUF files (.gguf)
- Any format supported by ComfyUI's model loaders

## Example Models on Hugging Face

### Upscale Models
- `Phips/2xNomosUni_span_multijpg` - High-quality 2x upscaler
- `ai-forever/Real-ESRGAN` - Popular Real-ESRGAN models

### GGUF Models
- `QuantStack/Qwen-Image-Edit-2509-GGUF` - Qwen Image Edit model (quantized)
- Search for more GGUF models on Hugging Face Hub!

## Changelog

### 2.1.0
- Added GGUF Model Loader From HF node
- Support for loading quantized GGUF models from Hugging Face
- Enhanced documentation with GGUF usage examples

### 2.0.0
- Renamed project to "Woohee HF Loader" for broader scope
- Preparing infrastructure for additional model loaders (checkpoints, LoRAs, etc.)
- Maintained backward compatibility with existing workflows

### 1.0.2
- Removed API token requirement (public repositories only)
- Added subfolder support for models in subdirectories
- Improved cache key handling for subfolder paths

### 1.0.1
- Bug fixes and improvements

### 1.0.0
- Initial release
- Support for loading upscale models from Hugging Face Hub
- Automatic model caching

## License

MIT

## Author

woohee5511

## Credits

Inspired by [ComfyUI-HfLoader](https://github.com/olduvai-jp/ComfyUI-HfLoader) by olduvai-jp
