# Skill Rationale

## Why this tool should exist

- Classification is the team's selected project category.
- Chest X-ray diagnosis is a practical target because ChatClinic already supports raster image and DICOM intake.
- TorchXRayVision provides open-source pretrained CXR classifiers with reproducible model names, labels, and preprocessing.
- The integration exposes multiple TorchXRayVision presets so users can compare all-data, dataset-specific, and higher-resolution CXR classifiers from the same UI.
- CheXbert and the CheXpert labeler provide open-source chest radiology report-labeling references with a standard 14-observation label convention.
- Pairing image classification with report-text labeling gives ChatClinic two complementary classification artifacts for the same clinical domain.

## Why the orchestrator should call it

- The tool adds clinically meaningful image-derived artifacts beyond generic image metadata.
- The report labeler adds clinically meaningful text-derived observation labels beyond generic text preview.
- It should run after intake because it needs a durable image or DICOM path.
- It should not run before source review, because source review establishes file type, preview state, and metadata.
- The report labeler should run after text review because text review establishes a durable text path and preview state.

## Why approval is or is not required

- Single-image CXR inference is low runtime and does not modify source data, so routine approval is not required.
- Batch inference or GPU-heavy workflows should request approval because they may consume substantial compute.

## What educational value it adds

- Demonstrates how to integrate an open-source medical imaging model into ChatClinic's plugin/runtime/Studio architecture.
- Demonstrates how one plugin can expose multiple open-source model presets while preserving provenance for each run.
- Demonstrates how to integrate a clinical text classification label set into the same plugin/runtime/Studio architecture.
- Shows the distinction between deterministic tool output and grounded LLM explanation.
- Forces explicit orchestration rules for when a medical AI classifier should and should not be used.
- Shows how image-derived probabilities and report-text-derived labels can be displayed side-by-side without conflating their evidence sources.

## Safety framing

- The tool produces model probabilities, not a diagnosis.
- The Studio card and grounded chat should preserve uncertainty, show warnings, and require human clinical review.
- The report labeler produces report-text-derived labels, not new image findings.
- The lightweight report backend follows the CheXpert label convention but is not the full CheXbert BERT checkpoint.
