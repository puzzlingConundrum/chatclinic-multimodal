# Skill Patch Proposal

## Tool

- `cxr_classification_tool`
- `cxr_report_labeling_tool`

## Purpose

- Run an open-source TorchXRayVision chest X-ray classifier preset on raster CXR images or DICOM chest radiographs.
- Produce model-derived pathology probabilities, top predictions, threshold flags, warnings, and provenance for Studio review.
- Label chest radiology report text with CheXbert/CheXpert-compatible observations.
- Produce report-text-derived positive, negative, uncertain, and blank labels for the same CXR observation family.

## When to use

- Use after `image_review_tool` for PNG/JPG/TIFF/BMP/WEBP sources that are likely chest radiographs.
- Use after `dicom_review_tool` for DICOM sources whose modality/context indicates a chest radiograph.
- Use when the user asks for CXR classification, CXR diagnosis draft, pneumonia/cardio-thoracic pathology screening, or `@cxr`.
- Use `model_weights` when the user asks to compare or select a specific open-source TorchXRayVision preset such as all-data, CheXpert, NIH, PadChest, RSNA, MIMIC, or ResNet.
- Use after `text_review_tool` when a text/markdown source is likely a chest radiology report or impression.
- Use when the user asks for report labels, CheXbert/CheXpert labels, CXR report classification, or `@cxrreport`.

## When not to use

- Do not use for CT, MRI, ultrasound, pathology slides, retinal images, or non-medical photographs.
- Do not use for NIfTI brain MRI dementia classification.
- Do not use report labeling for arbitrary clinical notes or non-chest reports.
- Do not present the output as a final clinical diagnosis.
- Do not invent pixel findings beyond the returned model probabilities.
- Do not present report labels as new image findings.

## Source type

- `image`
- `dicom`
- `text`

## Modality

- `medical-image`

## Recommended stage

- `post-intake`
- Run after the source review tool has persisted the uploaded image/DICOM path.

## Depends on

- `image_review_tool` for raster images
- `dicom_review_tool` for DICOM sources
- `text_review_tool` for CXR report text sources

## Runtime

- host compatible: both
- supported accelerators: cpu, cuda
- preferred accelerator: cuda
- requires gpu: no
- cpu fallback allowed: yes
- estimated runtime: 5-15 sec for one image after model weights are available
- first run of each model preset may download an additional TorchXRayVision checkpoint

## Approval policy

- approval not required for single-image inference
- approval recommended if running batch inference over many images

## Produces

- `cxr_classification_result`
- `artifacts.cxr_classification`
- Studio card: `cxr_classification`
- `model_preset`
- `available_model_weights`
- `cxr_report_labeling_result`
- `artifacts.cxr_report_labels`
- Studio card: `cxr_report_labels`

## Example routing note

- Use `image_review_tool` or `dicom_review_tool` first, then route to `cxr_classification_tool` only when the source is likely a chest radiograph. In grounded chat, describe the result as model-derived probability output and recommend clinician review.
- Use `text_review_tool` first, then route to `cxr_report_labeling_tool` when the text is likely a chest radiology report. In grounded chat, describe labels as report-text-derived observations and do not claim new pixel findings.
