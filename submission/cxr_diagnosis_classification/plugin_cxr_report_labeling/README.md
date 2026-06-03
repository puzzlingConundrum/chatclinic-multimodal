# CXR Report Labeling Tool

This plugin labels chest radiology report text with CheXbert/CheXpert-compatible observation labels.

## Background

The tool is designed as a lightweight companion to `cxr_classification_tool`:

- `cxr_classification_tool` classifies CXR pixels.
- `cxr_report_labeling_tool` classifies CXR report text.

It uses the 14-observation CheXpert/CheXbert label convention:

- `1`: positive
- `0`: negative
- `-1`: uncertain
- `null`: blank / not mentioned

The default backend is a deterministic local rule labeler. It follows the same label set and label convention, but it is not the full CheXbert BERT checkpoint.

## Inputs

```json
{
  "text_path": "/absolute/path/to/report.txt",
  "file_name": "report.txt"
}
```

Inline text is also supported:

```json
{
  "report_text": "FINDINGS: Mild cardiomegaly. No pneumothorax."
}
```

## Output

The result contains:

- `labels`: all 14 observation labels
- `positive_labels`
- `negative_labels`
- `uncertain_labels`
- `blank_labels`
- `likely_cxr_report`
- `warnings`
- `provenance`

The artifact slot is:

```text
artifacts.cxr_report_labels
```

Studio renderer:

```text
cxr_report_labels
```

Direct command aliases:

```text
@cxrreport
@chexbert
@chexpert
```

## References

- CheXbert: https://github.com/stanfordmlgroup/CheXbert
- CheXpert labeler: https://github.com/stanfordmlgroup/chexpert-labeler
- CheXbert paper: https://arxiv.org/abs/2004.09167
- CheXpert paper: https://arxiv.org/abs/1901.07031

## Clinical Framing

Labels are derived from report text only. They are not new image findings and are not a final clinical diagnosis.
