# CXR Diagnosis Classification Tools

KAIST AI619 ChatClinic Project

---

## 1. Project Goal

- Add chest X-ray diagnosis classification tools to ChatClinic
- Support raster CXR images and DICOM radiographs
- Support chest radiology report text
- Produce structured pathology probabilities
- Produce CheXbert/CheXpert-compatible report labels
- Render results in Studio
- Let `$studio` grounded chat explain tool-derived outputs

---

## 2. Motivation

- Team topic: Classification
- Selected task: CXR diagnosis classification
- Chest X-ray is a practical fit for ChatClinic because:
  - ChatClinic already supports PNG/JPG/TIFF image intake
  - ChatClinic already supports DICOM intake
  - ChatClinic already supports text/markdown intake
  - CXR classification can be represented as compact probability artifacts
  - CXR report labeling can be represented as compact observation-label artifacts

---

## 3. Open-Source Models

- Library: TorchXRayVision
- License: Apache-2.0
- Supported TorchXRayVision presets:

```python
xrv.models.DenseNet(weights="densenet121-res224-all", apply_sigmoid=True)
xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=True)
xrv.models.DenseNet(weights="densenet121-res224-nih", apply_sigmoid=True)
xrv.models.DenseNet(weights="densenet121-res224-pc", apply_sigmoid=True)
xrv.models.DenseNet(weights="densenet121-res224-rsna", apply_sigmoid=True)
xrv.models.DenseNet(weights="densenet121-res224-mimic_nb", apply_sigmoid=True)
xrv.models.DenseNet(weights="densenet121-res224-mimic_ch", apply_sigmoid=True)
xrv.models.ResNet(weights="resnet50-res512-all", apply_sigmoid=True)
```

- Output: pathology probabilities for standard CXR labels

Report-text tool:

- Reference: CheXbert / CheXpert labeler
- Current backend: lightweight deterministic CheXpert-compatible rules
- Output: positive, negative, uncertain, blank labels for 14 observations

---

## 4. Background References

- TorchXRayVision: open-source CXR datasets and models
- CheXNet: DenseNet-style pneumonia/CXR classification baseline
- CheXpert: large-scale CXR classification dataset context
- Cross-domain generalization paper: highlights calibration and domain-shift risk

---

## 5. Plugin Contract

Tool names:

```text
cxr_classification_tool
cxr_report_labeling_tool
```

Inputs:

```json
{
  "image_path": "...",
  "dicom_path": "...",
  "model_weights": "densenet121-res224-all | chex | nih | rsna | ...",
  "threshold": 0.5,
  "device": "auto"
}
```

Outputs:

- `artifacts.cxr_classification`
- `artifacts.cxr_report_labels`
- `top_predictions`
- `positive_findings`
- `model_preset`
- `available_model_weights`
- `positive_labels`
- `uncertain_labels`
- `probabilities`
- `warnings`
- `provenance`

---

## 6. Backend Integration

Image workflow:

```text
image_review_tool
  -> cxr_classification_tool
```

DICOM workflow:

```text
dicom_review_tool
  -> cxr_classification_tool
```

Text workflow:

```text
text_review_tool
  -> cxr_report_labeling_tool
```

The upload workflow remains robust:

- if dependencies are missing, return structured warning
- do not crash source upload
- preserve `used_tools`

---

## 7. Frontend Integration

Added Studio renderer:

```text
cxr_classification
cxr_report_labels
```

Studio card displays:

- model and device
- top finding
- top score
- threshold
- top predictions
- positive flags
- pathology probability bars
- available model presets
- provenance and warnings

Homepage/Sources panel:

- image/DICOM source: CXR model preset buttons
- text report source: CXR report-label button

---

## 8. Direct Chat Command

Added direct command:

```text
@cxr
@cxrreport
@chexbert
@chexpert
```

Behavior:

- uses active image or DICOM source
- calls `/api/v1/tools/cxr/run`
- updates source artifacts
- activates CXR Classification Studio card
- returns concise assistant summary
- for text reports, `@cxrreport` calls `/api/v1/tools/cxrreport/run`
- activates CXR Report Labels Studio card

---

## 9. Skill Patch Policy

Use the tool when:

- source is likely a chest radiograph
- user asks for CXR classification, CXR diagnosis draft, pathology probabilities, pneumonia screening, or `@cxr`
- text is likely a chest radiology report and user asks for report labels, CheXbert/CheXpert labels, or `@cxrreport`

Do not use the tool for:

- CT
- MRI
- NIfTI brain volumes
- ultrasound
- pathology slides
- non-medical images
- arbitrary clinical notes
- non-chest reports

---

## 10. Safety Framing

The tool output is:

- model-derived probability output
- report-text-derived observation labeling
- research/screening support
- not a final clinical diagnosis

Grounded chat must:

- preserve warnings
- avoid inventing pixel findings
- avoid presenting report labels as new image findings
- recommend clinician review

---

## 11. Runtime and Checkpoints

Runtime:

- CPU supported
- CUDA supported when available
- single-image inference tested on CPU

Checkpoint/cache path:

```text
checkpoints/torchxrayvision/
```

Observed weight:

```text
nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
```

---

## 12. Verification

Passed:

- Python compile check
- TorchXRayVision import
- PNG plugin inference
- DICOM plugin inference
- image workflow artifact attachment
- DICOM workflow artifact attachment
- report-text plugin inference
- text workflow report-label artifact attachment
- backend `/api/v1/tools/cxr/run`
- image upload endpoint
- frontend dev server in user terminal
- frontend production build

Pending:

- optional test with external real CXR DICOM

---

## 13. Demo Plan

1. Start backend on port 8001
2. Start frontend on port 3000
3. Upload `examples/cxr/torchxrayvision_00000001_000.png`
4. Use the Sources-panel CXR launcher to run All-data and CheXpert presets
5. Open Studio
6. Show `Image Review`
7. Show `CXR Classification`
8. Upload `examples/cxr/sample_cxr_report.txt`
9. Show `CXR Report Labels`
10. Ask `$studio CXR report labels 결과를 설명해줘`
11. Run `@cxr` and `@cxrreport` on active sources

---

## 14. Limitations

- Probability scores are not calibrated clinical diagnoses
- Cross-site/domain shift may affect performance
- Demo DICOM is generated from a public PNG for software testing
- Demo report text is synthetic and de-identified for software testing
- Tool should not be applied outside chest radiographs

---

## 15. Conclusion

- Added working CXR image and report classification tools to ChatClinic
- Added multiple CXR image model presets and homepage launcher controls
- Integrated plugin runtime, workflow, Studio, direct chat, and Skill rules for both tools
- Included examples, references, checkpoint instructions, and test logs
- Submission package is ready for PR review and presentation preparation
