# Test Log

Environment:

```text
conda env: med
python: 3.12.13
torch: 2.11.0+cu130
torchvision: 0.26.0+cu130
torchxrayvision: 1.4.0
CUDA availability in local test: false
```

## Dependency Import

Command:

```bash
conda run -n med python -c "import torch, torchvision, torchxrayvision as xrv, pydicom, skimage; print('ok')"
```

Result:

```text
ok
```

## Python Compile Check

Command:

```bash
conda run -n med python -m py_compile app/main.py app/models.py app/services/*.py plugins/cxr_classification_tool/logic.py
```

Result:

```text
passed
```

## Plugin Test: PNG

Input:

```text
examples/cxr/torchxrayvision_00000001_000.png
```

Result:

```text
status: ok
top finding: Cardiomegaly (0.773514)
top 3:
  - Cardiomegaly: 0.773514
  - Fibrosis: 0.758762
  - Emphysema: 0.750558
```

## Plugin Test: Alternate CXR Model Preset

Input:

```text
examples/cxr/torchxrayvision_00000001_000.png
```

Command model option:

```text
model_weights: chex
```

Result:

```text
status: ok
model: densenet121-res224-chex
model preset: DenseNet121, CheXpert, 224px
probability rows: 11
raw output count: 18
omitted untrained outputs: 7
top 2:
  - Cardiomegaly: 0.788684
  - Enlarged Cardiomediastinum: 0.748425
available model presets: 8
```

## Plugin Test: Demo DICOM

Input:

```text
examples/cxr/torchxrayvision_00000001_000_demo.dcm
```

Result:

```text
status: ok
top finding: Cardiomegaly (0.773514)
top 3:
  - Cardiomegaly: 0.773514
  - Fibrosis: 0.758762
  - Emphysema: 0.750558
```

## Image Workflow Test

Input:

```text
examples/cxr/torchxrayvision_00000001_000.png
```

Result:

```text
cxr_classification status: ok
cxr_classification Studio card: true
used_tools:
  - image_review_tool
  - cxr_classification_tool
```

## DICOM Workflow Test

Input:

```text
examples/cxr/torchxrayvision_00000001_000_demo.dcm
```

Result:

```text
cxr_classification status: ok
cxr_classification Studio card: true
used_tools:
  - dicom_review_tool
  - cxr_classification_tool
```

## Plugin Test: CXR Report Text

Input:

```text
examples/cxr/sample_cxr_report.txt
```

Result:

```text
status: ok
positive labels:
  - Cardiomegaly
  - Atelectasis
  - Support Devices
negative labels include:
  - Edema
  - Consolidation
  - Pneumothorax
  - Pleural Effusion
```

## Text Workflow Test

Input:

```text
examples/cxr/sample_cxr_report.txt
```

Result:

```text
cxr_report_labels status: ok
cxr_report_labels Studio card: true
used_tools:
  - text_review_tool
  - cxr_report_labeling_tool
```

## Report Labeling Alias / Context Test

Alias resolver:

```text
cxrreport -> cxr_report_labeling_tool
chexbert -> cxr_report_labeling_tool
```

Grounded text context:

```text
cxr_report_labels present: true
status: ok
```

## Backend HTTP Test

Endpoint:

```text
POST /api/v1/tools/cxr/run
```

Result:

```text
tool_name: cxr_classification_tool
status: ok
studio.renderer: cxr_classification
```

## Upload Workflow HTTP Test

Endpoint:

```text
POST /api/v1/image/upload
```

Result:

```text
used_tools:
  - image_review_tool
  - cxr_classification_tool
studio_cards:
  - image_review
  - cxr_classification
```

## Frontend

Observed:

```text
npm ci: succeeded in the user's activated terminal
npm run dev: Next.js server ready at http://localhost:3000
npm run build: succeeded
Homepage/Sources CXR launcher: added model preset buttons for image/DICOM and report-label button for text
```

Build note:

```text
Next.js emitted a workspace-root warning because both the repository root and webapp directory contain lockfiles.
The build still completed successfully.
```
