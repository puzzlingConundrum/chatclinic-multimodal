# Pull Request Description

## Title

Add CXR image classification and report labeling tools

## Summary

This PR integrates two related chest X-ray classification tools into ChatClinic.

The new `cxr_classification_tool` uses TorchXRayVision DenseNet/ResNet pretrained presets to produce pathology probability outputs for chest radiograph images and DICOM sources. Results are attached to source analysis as `artifacts.cxr_classification`, rendered in Studio with a new `CXR Classification` card, and made available to grounded chat.

The new `cxr_report_labeling_tool` labels chest radiology report text with CheXbert/CheXpert-compatible 14-observation labels. Results are attached as `artifacts.cxr_report_labels`, rendered in Studio with a new `CXR Report Labels` card, and made available to grounded text chat.

## Main Changes

- Added `plugins/cxr_classification_tool`
  - `tool.json`
  - `logic.py`
  - `README.md`
  - `requirements.txt`
- Added `plugins/cxr_report_labeling_tool`
  - `tool.json`
  - `logic.py`
  - `README.md`
- Updated image and DICOM workflows to run CXR classification after source review
- Updated text workflow to attach CXR report labels when text looks like a chest radiology report
- Added CXR classification artifact support to grounded image/DICOM chat context
- Added CXR report label artifact support to grounded text chat context
- Added frontend Studio renderer for CXR pathology probabilities
- Added frontend Studio renderer for CXR report labels
- Added direct `@cxr` execution for active image/DICOM sources
- Added homepage/Sources-panel CXR launcher with multiple TorchXRayVision model preset buttons
- Added direct `@cxrreport`, `@chexbert`, and `@chexpert` execution for active text sources
- Updated master Skill routing guidance
- Added example PNG, demo DICOM, and sample CXR report files for testing
- Added submission documents, references, and slide draft
- Added TorchXRayVision-related dependencies to `requirements.txt` and `environment.yml`

## Safety / Clinical Framing

- The tool output is model-derived probability output, not a final clinical diagnosis.
- The tool should only be used for chest radiographs.
- The report labeler output is report-text-derived observation labeling, not new image findings.
- The report labeler should only be used for chest radiology reports/impressions.
- Non-CXR images, CT, MRI, NIfTI, ultrasound, pathology images, and non-medical photos are out of scope.
- Grounded chat should describe probabilities and warnings without inventing pixel findings.

## Runtime

- CPU supported
- CUDA supported when available
- Tested locally in conda env `med`
- Default checkpoint cache path:

```text
checkpoints/torchxrayvision/
```

## Verification

- Python compile check passed
- TorchXRayVision dependency import passed
- Direct plugin test passed for PNG
- Direct plugin test passed for alternate CheXpert CXR model preset
- Direct plugin test passed for demo DICOM
- Direct report-labeling plugin test passed for sample CXR report text
- Image workflow attaches CXR classification artifact
- DICOM workflow attaches CXR classification artifact
- Text workflow attaches CXR report label artifact
- Running backend `/api/v1/tools/cxr/run` returns `status: ok`
- Running image upload endpoint returns both `image_review_tool` and `cxr_classification_tool`
- Frontend `npm run build` passed
- Homepage/Sources-panel launcher supports image/DICOM preset selection and report-labeling execution

## Known Follow-ups

- Upload a real external CXR DICOM if a licensed sample is provided by the instructor/team
- Optionally export `slides/cxr_diagnosis_classification_presentation.md` to PDF for final submission
