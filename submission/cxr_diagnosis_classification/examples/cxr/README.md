# CXR Example Files

This directory contains small example files for testing the CXR classification workflow.

## Files

- `torchxrayvision_00000001_000.png`
  - Source: TorchXRayVision test image
  - URL: https://raw.githubusercontent.com/mlmed/torchxrayvision/main/tests/00000001_000.png
  - Project: https://github.com/mlmed/torchxrayvision
  - License: Apache-2.0, matching the TorchXRayVision repository

- `torchxrayvision_00000001_000_demo.dcm`
  - Derived demo DICOM generated locally from `torchxrayvision_00000001_000.png`
  - Generator: `make_demo_dicom.py`
  - Purpose: tests ChatClinic DICOM upload, DICOM review, and CXR classification routing
  - Note: this is not an original clinical DICOM export; it is a de-identified demo DICOM wrapper around the public PNG pixels.

- `sample_cxr_report.txt`
  - Synthetic de-identified chest radiology report text
  - Purpose: tests Text Review and `cxr_report_labeling_tool`
  - Expected behavior: Studio shows a `CXR Report Labels` card with CheXbert/CheXpert-compatible observations.

## Regenerate Demo DICOM

```bash
conda activate med
python examples/cxr/make_demo_dicom.py
```

## Quick Tests

PNG:

```bash
conda activate med
python -c "from plugins.cxr_classification_tool.logic import run; print(run({'image_path':'examples/cxr/torchxrayvision_00000001_000.png','device':'cpu','top_k':3})['summary'])"
```

DICOM:

```bash
conda activate med
python -c "from plugins.cxr_classification_tool.logic import run; print(run({'dicom_path':'examples/cxr/torchxrayvision_00000001_000_demo.dcm','device':'cpu','top_k':3})['summary'])"
```

Report text:

```bash
conda activate med
python -c "from plugins.cxr_report_labeling_tool.logic import run; print(run({'text_path':'examples/cxr/sample_cxr_report.txt'})['summary'])"
```

These files are for software testing and demonstration only. Model outputs are screening/research support and are not final clinical diagnoses.
