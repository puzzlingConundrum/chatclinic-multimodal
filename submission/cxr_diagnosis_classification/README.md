# CXR Diagnosis Classification Submission

## Project Metadata

- Project name: `cxr_diagnosis_classification`
- Topic category: Classification
- Tool name: `cxr_classification_tool`
- Additional tool name: `cxr_report_labeling_tool`
- Modality: medical image
- Additional modality: clinical text
- Supported source types: PNG/JPG/TIFF-style raster image, DICOM, text/markdown report
- Main open-source model/library: TorchXRayVision, CheXbert/CheXpert labeler references
- Default model weights: `densenet121-res224-all`
- Supported image model presets: `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-nih`, `densenet121-res224-pc`, `densenet121-res224-rsna`, `densenet121-res224-mimic_nb`, `densenet121-res224-mimic_ch`, `resnet50-res512-all`
- Runtime: CPU or CUDA GPU, with CPU fallback
- Approval required: no for single-image inference

## Summary

This submission adds two related chest X-ray classification tools to ChatClinic.

The first tool runs open-source TorchXRayVision DenseNet/ResNet classifier presets on a chest radiograph and returns model-derived pathology probabilities. The result is attached to ChatClinic as `artifacts.cxr_classification`, displayed in Studio as a `CXR Classification` card, and made available to grounded `$studio` chat.

The second tool labels chest radiology report text using CheXbert/CheXpert-compatible observation labels. The result is attached as `artifacts.cxr_report_labels`, displayed as a `CXR Report Labels` card, and made available to grounded text chat.

The output is research/screening support only and must not be presented as a final clinical diagnosis.

## Submitted Files

```text
cxr_diagnosis_classification/
  plugin/
    tool.json
    logic.py
    README.md
    requirements.txt
  plugin_cxr_report_labeling/
    tool.json
    logic.py
    README.md
  skill_update/
    skill_patch.md
    skill_rationale.md
  references/
    background_papers.md
  slides/
    cxr_diagnosis_classification_presentation.md
  examples/
    cxr/
      README.md
      make_demo_dicom.py
      torchxrayvision_00000001_000.png
      torchxrayvision_00000001_000_demo.dcm
      sample_cxr_report.txt
  checkpoints/
    README.md
  PR_DESCRIPTION.md
  TEST_LOG.md
```

## ChatClinic Integration

Implemented integration points in the platform repository:

- `plugins/cxr_classification_tool/`
  - new plugin package and manifest
- `plugins/cxr_report_labeling_tool/`
  - new report labeling plugin package and manifest
- `app/services/workflows.py`
  - image and DICOM workflows attach `artifacts.cxr_classification`
  - text workflows attach `artifacts.cxr_report_labels` when the text looks like a CXR report
  - upload still succeeds if optional classifier dependencies are missing
- `app/services/chat.py`
  - grounded image/DICOM chat includes CXR classification artifact context
  - grounded text chat includes CXR report label artifact context
- `webapp/app/components/customStudioRenderers.tsx`
  - Studio `CXR Classification` renderer
  - Studio `CXR Report Labels` renderer
- `webapp/app/page.tsx`
  - direct `@cxr` execution for active image/DICOM sources
  - homepage/Sources-panel CXR launcher with multiple model preset buttons
  - direct `@cxrreport`, `@chexbert`, and `@chexpert` execution for active text sources
- `skills/chatgenome-orchestrator/SKILL.md`
  - orchestration guidance for when to use and not use the tool
- `requirements.txt` and `environment.yml`
  - optional TorchXRayVision runtime dependencies

If this zip is reviewed as a standalone plugin package, copy the `plugin/` directory contents into:

```text
chatclinic-multimodal/plugins/cxr_classification_tool/
```

The included `tool.json` entrypoint expects that platform path:

```text
plugins.cxr_classification_tool.logic:execute
```

Copy `plugin_cxr_report_labeling/` contents into:

```text
chatclinic-multimodal/plugins/cxr_report_labeling_tool/
```

Its `tool.json` entrypoint expects:

```text
plugins.cxr_report_labeling_tool.logic:execute
```

## Environment Setup

Use the repository `environment.yml` as the base environment.

Additional runtime dependencies are listed in:

```text
plugin/requirements.txt
```

Installed and tested in the local `med` conda environment:

```bash
conda activate med
pip install "torchxrayvision>=1.4,<2.0" "scikit-image>=0.23,<1.0" "pydicom>=2.4,<4.0"
```

## Checkpoint Setup

Default checkpoint/cache path:

```text
chatclinic-multimodal/checkpoints/torchxrayvision/
```

TorchXRayVision downloads the pretrained weight file on first use if it is not already present. For offline submission, place the checkpoint at the path above or set:

```bash
export TORCHXRAYVISION_CACHE_DIR=/path/to/chatclinic-multimodal/checkpoints/torchxrayvision
```

See `checkpoints/README.md` for the exact filename observed in local testing.

## Quick Verification

From `chatclinic-multimodal`:

```bash
conda activate med
python -m py_compile app/main.py app/models.py app/services/*.py plugins/cxr_classification_tool/logic.py
```

Plugin smoke test:

```bash
python -c "from plugins.cxr_classification_tool.logic import run; print(run({'image_path':'examples/cxr/torchxrayvision_00000001_000.png','device':'cpu','top_k':3})['summary'])"
```

Alternate model preset smoke test:

```bash
python -c "from plugins.cxr_classification_tool.logic import run; print(run({'image_path':'examples/cxr/torchxrayvision_00000001_000.png','model_weights':'chex','device':'cpu','top_k':2})['summary'])"
```

DICOM smoke test:

```bash
python -c "from plugins.cxr_classification_tool.logic import run; print(run({'dicom_path':'examples/cxr/torchxrayvision_00000001_000_demo.dcm','device':'cpu','top_k':3})['summary'])"
```

Report label smoke test:

```bash
python -c "from plugins.cxr_report_labeling_tool.logic import run; print(run({'text_path':'examples/cxr/sample_cxr_report.txt'})['summary'])"
```

Backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Frontend:

```bash
cd webapp
npm ci
npm run dev
```

Open `http://localhost:3000` and upload either example file under `examples/cxr/`.
For report labeling, upload `examples/cxr/sample_cxr_report.txt`.

## Expected Result

After uploading a CXR PNG or DICOM:

- `Image Review` or `DICOM Review` card appears
- `CXR Classification` Studio card appears
- pathology probabilities and top predictions are shown
- selected model preset metadata and available preset names are shown
- `used_tools` includes `cxr_classification_tool`
- the homepage/Sources-panel launcher can rerun `@cxr` with different TorchXRayVision presets
- `$studio` chat can explain the CXR classification artifact as model-derived probabilities
- for report text, `Text Review` and `CXR Report Labels` cards appear
- report labels are shown as positive, negative, uncertain, or blank
- `used_tools` includes `cxr_report_labeling_tool`

## Limitations

- This is not a clinical diagnosis system.
- The classifier should only be used for chest radiographs.
- The report labeler should only be used for chest radiology report text.
- Non-CXR images may produce misleading probabilities.
- Report labels are text-derived observations, not new image findings.
- Site/domain shift can affect CXR model calibration and generalization.
- GPU is optional for single-image inference, but CUDA is preferred when available.
