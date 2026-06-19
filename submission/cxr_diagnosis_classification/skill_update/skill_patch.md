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

---

# Skill Patch Addendum — two-stage ensemble with LLM escalation

## Tools

- `cxr_ensemble_tool` (two-stage orchestrator; the recommended CXR entry point)
- `cxr_classifier_tool` (MedCLIP / BiomedCLIP zero-shot model, used inside Stage 2)

## Purpose

- Improve reliability over a single CXR model by adding a confidence-aware second stage.
- **Stage 1** runs DenseNet-121 and reports raw confidence (top score + margin).
- An **escalation controller reads those raw scores** and decides whether one model is
  enough or whether to escalate. By default an LLM (`gpt-5-mini`) makes this decision,
  replacing hard-coded thresholds; a deterministic threshold rule is the fallback.
- **Stage 2** adds ResNet-50 + MedCLIP, aligns the 18/14-label vocabularies to a canonical
  set, and fuses them via weighted score fusion.

## When to use

- Prefer `cxr_ensemble_tool` over the single-model `cxr_classification_tool` whenever a more
  robust, second-opinion CXR result is wanted, or whenever the primary result may be
  low-confidence or ambiguous.
- Use `decision_mode=threshold` to force the legacy hard-coded rule (e.g. offline, no API
  key, or to demonstrate the old-vs-new comparison). Use `decision_mode=llm` to force the
  LLM router. Default `auto` uses the LLM when `OPENAI_API_KEY` is set, else the threshold.
- Use `stage=1` to run DenseNet only (the explicit staged UI: show Stage 1, then let the
  user trigger Stage 2). Use `stage=2` to run the decision + ensemble on demand.

## When not to use

- Same exclusions as `cxr_classification_tool`: not for CT/MRI/NIfTI/ultrasound/pathology or
  non-CXR images.
- Do not present fused output as a final clinical diagnosis.

## Routing decision input

- The escalation decision keys off the **Stage-1 raw confidence scores** (top finding, top
  score, top-1-vs-top-2 margin, top-5), not the raw pixels.

## Produces

- `cxr_ensemble_result` / `artifacts.cxr_ensemble`
- Studio card: `cxr_ensemble` (shows Stage, escalation decision + `decided_by`, fused
  findings, and explicit "Run Stage 2 · LLM / Threshold" controls)
- `decision { mode, escalate, reason, raw_confidence, llm_model, threshold_would_be }`

## Runtime

- No OpenAI key required for inference: `auto` falls back to the threshold rule offline.
- CPU or CUDA; single RTX 3090 sufficient (≤4 supported). MedCLIP/BiomedCLIP weights
  auto-download from HuggingFace on first use.
