# CXR Report Labeling Tool Integration Plan

## Goal

Add a second classification tool that works on chest radiology report text.

The tool complements the existing CXR image classifier:

- `cxr_classification_tool`: classifies CXR image pixels
- `cxr_report_labeling_tool`: classifies CXR report text into observation labels

## Open-Source Reference

Primary reference:

- CheXbert: https://github.com/stanfordmlgroup/CheXbert
- Paper: https://arxiv.org/abs/2004.09167

Related reference:

- CheXpert labeler: https://github.com/stanfordmlgroup/chexpert-labeler
- CheXpert paper: https://arxiv.org/abs/1901.07031

## Implementation Choice

The current integration uses a lightweight deterministic backend that follows the CheXbert/CheXpert 14-observation label set and label convention.

Label convention:

- positive: `1`
- negative: `0`
- uncertain: `-1`
- blank: `null`

This avoids adding a large BERT checkpoint to the runtime while still providing a useful report-text classification artifact. The full CheXbert checkpoint can be added later as an optional backend.

## Integration Steps

Implemented:

- Added `plugins/cxr_report_labeling_tool`
- Added automatic text workflow attachment for likely CXR reports
- Added `artifacts.cxr_report_labels`
- Added Studio card `cxr_report_labels`
- Added direct aliases:
  - `@cxrreport`
  - `@chexbert`
  - `@chexpert`
- Added grounded text-chat context for report labels
- Added example report:
  - `examples/cxr/sample_cxr_report.txt`

## Safety Policy

- Use only for chest radiology reports or impressions.
- Do not use for arbitrary clinical notes or non-chest reports.
- Labels are report-text-derived observations, not new image findings.
- The output is not a final clinical diagnosis.

## Verification

Completed checks:

```bash
conda run -n med python -m py_compile app/main.py app/models.py app/services/*.py plugins/cxr_report_labeling_tool/logic.py plugins/cxr_classification_tool/logic.py
conda run -n med python -c "from plugins.cxr_report_labeling_tool.logic import run; print(run({'text_path':'examples/cxr/sample_cxr_report.txt'})['summary'])"
```

Observed result:

```text
status: ok
positive labels: Cardiomegaly, Atelectasis, Support Devices
```

Workflow:

```text
text_review_tool -> cxr_report_labeling_tool
Studio card: cxr_report_labels
```

Frontend build:

```bash
cd webapp
npm run build
```

Result:

```text
passed
```
