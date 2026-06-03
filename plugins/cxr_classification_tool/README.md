# CXR Classification Tool

This plugin runs open-source TorchXRayVision chest X-ray pathology classifiers and returns pathology probabilities for Studio rendering.

## Inputs

```json
{
  "image_path": "/absolute/path/to/chest_xray.png",
  "file_name": "chest_xray.png",
  "model_weights": "densenet121-res224-all",
  "threshold": 0.5,
  "device": "auto"
}
```

For DICOM inputs, use `dicom_path` instead of `image_path`.

## Model Presets

Supported `model_weights` values:

- `densenet121-res224-all`: default all-data DenseNet121 preset
- `densenet121-res224-chex`: CheXpert DenseNet121 preset
- `densenet121-res224-nih`: NIH ChestX-ray14 DenseNet121 preset
- `densenet121-res224-pc`: PadChest DenseNet121 preset
- `densenet121-res224-rsna`: RSNA Pneumonia DenseNet121 preset
- `densenet121-res224-mimic_nb`: MIMIC-CXR NIH-label DenseNet121 preset
- `densenet121-res224-mimic_ch`: MIMIC-CXR CheXpert-label DenseNet121 preset
- `resnet50-res512-all`: higher-resolution 512px ResNet50 preset

Short aliases such as `all`, `chex`, `nih`, `pc`, `rsna`, `mimic_nb`, `mimic_ch`, and `resnet` are also accepted.

For dataset-specific TorchXRayVision presets, some raw outputs do not have trained labels. The plugin omits those unlabeled outputs from `probabilities` and records the omitted count in `omitted_untrained_output_count`.

Direct chat examples:

```text
@cxr
@cxr model_weights=chex threshold=0.4
@cxr model_weights=rsna
@cxr model_weights=resnet50-res512-all threshold=0.45
```

## Runtime Setup

Install the optional plugin dependencies:

```bash
pip install -r plugins/cxr_classification_tool/requirements.txt
```

Default checkpoint/cache path:

```text
checkpoints/torchxrayvision/
```

The plugin uses that repo-local path unless `TORCHXRAYVISION_CACHE_DIR` is set. Set this environment variable when using another submitted/offline weights directory:

```bash
export TORCHXRAYVISION_CACHE_DIR=/path/to/chatclinic-multimodal/checkpoints/torchxrayvision
```

If TorchXRayVision or weights are not available, the plugin returns a structured warning artifact instead of crashing the upload workflow.

## Output

The result contains:

- `available`: whether inference ran
- `status`: `ok`, `dependency_missing`, or `model_unavailable`
- `probabilities`: pathology probability rows
- `top_predictions`: highest-score pathology rows
- `model_preset`: selected architecture, source dataset, and native resolution metadata
- `available_model_weights`: supported TorchXRayVision model preset names
- `warnings`: runtime/model warnings
- `provenance`: model, source, preprocessing, device, and cache details

This is research/screening support only and not a final clinical diagnosis.
