# Woohee HF Upscaler Loader

Load upscale models directly from Hugging Face Hub in ComfyUI without manual downloads.

## Description

This custom node allows you to load upscale models (ESRGAN, Real-ESRGAN, etc.) directly from Hugging Face repositories into ComfyUI. No need to manually download and place models in your models folder - just provide the repository name and filename!

## Features

- Load upscale models directly from Hugging Face Hub
- Automatic model caching to avoid repeated downloads
- Support for both public and private repositories (with API token)
- Compatible with standard ComfyUI upscale model format
- Works with ImageUpscaleWithModel node

## Installation

### Via ComfyUI Manager

Search for "Woohee HF Upscaler Loader" in ComfyUI Manager and install.

### Manual Installation

1. Navigate to your ComfyUI custom_nodes directory
2. Clone this repository:
   ```bash
   git clone https://github.com/hw5511/comfyui_hf_upscaler_loader.git
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Restart ComfyUI

## Usage

1. Add "Upscale Model Loader From HF" node to your workflow
2. Enter the Hugging Face repository name (e.g., `Phips/2xNomosUni_span_multijpg`)
3. Enter the model filename (e.g., `2xNomosUni_span_multijpg.pth`)
4. (Optional) Enter subfolder path if the model is in a subdirectory
5. Connect the output to an "Upscale Image (using Model)" node

## Inputs

- **repo_name** (STRING): Hugging Face repository name (e.g., `Phips/2xNomosUni_span_multijpg`)
- **filename** (STRING): Model file name in the repository (e.g., `2xNomosUni_span_multijpg.pth`)
- **subfolder** (STRING, optional): Subfolder path within the repository (e.g., `weights/v2`)

## Outputs

- **UPSCALE_MODEL**: Loaded upscale model compatible with ComfyUI's upscale nodes

## Example Workflow

```
Load Image -> Upscale Model Loader From HF -> Upscale Image (using Model) -> Save Image
```

## Supported Model Formats

- PyTorch models (.pth, .pt)
- SafeTensors (.safetensors)
- Any format supported by ComfyUI's upscale model loader

## Example Models on Hugging Face

- `Phips/2xNomosUni_span_multijpg` - High-quality 2x upscaler
- `ai-forever/Real-ESRGAN` - Popular Real-ESRGAN models
- And many more on Hugging Face Hub!

## Changelog

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
