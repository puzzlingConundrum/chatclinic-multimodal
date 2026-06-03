# CXR Classification Tool Integration Plan

## Goal

ChatClinic에 chest X-ray diagnosis classification tool을 추가한다.

최종 목표는 사용자가 PNG/JPG/TIFF 또는 DICOM chest X-ray를 업로드했을 때:

- 기존 Image Review 또는 DICOM Review가 먼저 실행되고
- CXR classification plugin이 pathology probability를 생성하고
- Studio에서 CXR Classification card로 결과를 확인하고
- `$studio` grounded chat이 classification 결과를 설명할 수 있게 하는 것이다.

## Open-source Model Choice

1차 후보는 TorchXRayVision이다.

- Repository: https://github.com/mlmed/torchxrayvision
- Documentation: https://mlmed.org/torchxrayvision/models.html
- Paper: https://arxiv.org/abs/2111.00595
- License: Apache-2.0

선택 이유:

- Chest X-ray 전용 open-source library이다.
- 사전학습 DenseNet/ResNet classifier를 제공한다.
- 표준 pathology target list가 이미 정의되어 있다.
- PyTorch 기반이라 현재 `environment.yml`의 torch/torchvision stack과 잘 맞는다.
- CXR classification baseline으로 설명하기 쉽고, references/slides 작성도 수월하다.

지원 모델:

- `xrv.models.DenseNet(weights="densenet121-res224-all", apply_sigmoid=True)`
- `xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=True)`
- `xrv.models.DenseNet(weights="densenet121-res224-nih", apply_sigmoid=True)`
- `xrv.models.DenseNet(weights="densenet121-res224-pc", apply_sigmoid=True)`
- `xrv.models.DenseNet(weights="densenet121-res224-rsna", apply_sigmoid=True)`
- `xrv.models.DenseNet(weights="densenet121-res224-mimic_nb", apply_sigmoid=True)`
- `xrv.models.DenseNet(weights="densenet121-res224-mimic_ch", apply_sigmoid=True)`
- `xrv.models.ResNet(weights="resnet50-res512-all", apply_sigmoid=True)`

기본 모델:

- `densenet121-res224-all`

프론트엔드 homepage/Sources panel에서는 CXR image/DICOM source가 active일 때 위 모델 프리셋을 버튼으로 선택해 `@cxr model_weights=<preset>`를 실행한다. CXR report text source가 active일 때는 `@cxrreport` report-labeling button을 보여준다.

예상 output labels:

- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural Thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia
- Lung Lesion
- Fracture
- Lung Opacity
- Enlarged Cardiomediastinum

## Runtime Policy

TorchXRayVision와 pretrained weights는 제출/실행 환경에 준비되어야 한다.

권장 설치:

```bash
pip install torchxrayvision
```

체크포인트 제출 방식:

- TorchXRayVision pretrained weights는 첫 로딩 시 cache에 저장된다.
- 제출 시 Google Drive/Dropbox 또는 assignment system에 weights/cache archive를 제출한다.
- 권장 배치 경로:

```text
chatclinic-multimodal/checkpoints/torchxrayvision/
```

환경 변수:

```bash
export TORCHXRAYVISION_CACHE_DIR=/path/to/chatclinic-multimodal/checkpoints/torchxrayvision
```

GPU 정책:

- GPU가 있으면 CUDA 사용
- GPU가 없으면 CPU fallback 허용
- RTX 3090 최대 4장 환경에서는 single-image inference이므로 1 GPU만 사용해도 충분하다.

## Implementation Steps

### Step 1. Add plugin

Create:

```text
plugins/cxr_classification_tool/
  tool.json
  logic.py
  README.md
  requirements.txt
```

Plugin contract:

- accepts `image_path` or `dicom_path`
- optional `file_name`
- optional `model_weights`
  - supported values: `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-nih`, `densenet121-res224-pc`, `densenet121-res224-rsna`, `densenet121-res224-mimic_nb`, `densenet121-res224-mimic_ch`, `resnet50-res512-all`
  - short aliases: `all`, `chex`, `nih`, `pc`, `rsna`, `mimic_nb`, `mimic_ch`, `resnet`
- returns:
  - `tool`
  - `summary`
  - `result`
  - `artifacts`
  - `warnings`
  - `provenance`

### Step 2. Backend workflow integration

Update `app/services/workflows.py`:

- after `analyze_image_source`, call `run_tool("cxr_classification_tool", ...)`
- after `analyze_dicom_source`, call `run_tool("cxr_classification_tool", ...)`
- attach result as `analysis.artifacts["cxr_classification"]`
- add Studio card:

```json
{"id": "cxr_classification", "title": "CXR Classification", "subtitle": "Pathology probabilities"}
```

- append `cxr_classification_tool` to `used_tools`
- do not fail the upload if classification dependency/model loading fails; preserve warning instead.

### Step 3. Frontend Studio card

Update:

- `webapp/app/components/studioRenderers.tsx`
- `webapp/app/components/customStudioRenderers.tsx`

Add renderer key:

```text
cxr_classification
```

The card should show:

- model name
- model preset metadata
- device
- top prediction
- probability list
- warnings
- provenance

### Step 4. Skill update

Create/update:

```text
skill_update/cxr_classification_skill_patch.md
skill_update/cxr_classification_skill_rationale.md
references/background_papers.md
```

Skill policy:

- use after image/DICOM review
- use only for likely chest radiographs
- do not use for CT/MRI/NIfTI/non-CXR images
- results are screening/research support, not final diagnosis
- human clinician review required

### Step 5. Verification

Backend:

```bash
python -m py_compile app/main.py app/models.py app/services/*.py plugins/cxr_classification_tool/logic.py
```

Frontend:

```bash
cd webapp
npm run build
```

Manual:

1. start backend
2. start frontend
3. upload a CXR PNG/JPG
4. confirm Image Review card appears
5. confirm CXR Classification card appears
6. ask `$studio CXR classification 결과 설명해줘`

## Risks

- TorchXRayVision package may not be installed in the local environment.
- Pretrained weights may require download unless checkpoint cache is submitted.
- DICOM preprocessing depends on pydicom/pixel data availability.
- The classifier is not a clinical diagnosis system; outputs must be framed as model-derived probabilities.

## Current Scope

This integration starts with automatic classification for image and DICOM sources.

NIfTI brain MRI dementia classification is intentionally out of scope for the first version because it requires different preprocessing, a different model family, and stronger validation.

## Progress Log

Implemented:

- Added `plugins/cxr_classification_tool` with TorchXRayVision DenseNet inference.
- Added optional dependency entries to `requirements.txt` and `environment.yml`.
- Connected image and DICOM workflows so CXR classification is attached as `artifacts.cxr_classification`.
- Added Studio renderer metadata and a frontend CXR Classification card.
- Added direct `@cxr` execution for active image/DICOM sources.
- Added multiple TorchXRayVision model presets and short aliases.
- Added homepage/Sources-panel CXR launcher for image/DICOM model selection and report-text labeling.
- Updated grounded chat compaction so `$studio` can explain CXR classification artifacts.
- Updated the ChatGenome orchestrator skill with `@cxr` routing and safety language.
- Added submission support docs under `skill_update/` and `references/`.

Verified:

- `python3 -m py_compile app/main.py app/models.py app/services/*.py plugins/cxr_classification_tool/logic.py`
- Dependency-missing smoke test returns a structured warning instead of crashing when `torch`/TorchXRayVision are not installed.
- `med` conda environment imports TorchXRayVision, pydicom, scikit-image, torch, and torchvision.
- TorchXRayVision weights download/cache successfully under `checkpoints/torchxrayvision/`.
- Direct CXR plugin smoke test returns `status: ok`.
- Image workflow attaches `artifacts.cxr_classification`, adds the `cxr_classification` Studio card, and records `cxr_classification_tool`.
- Running backend endpoint `/api/v1/tools/cxr/run` returns `status: ok`.
- Running image upload endpoint `/api/v1/image/upload` returns both `image_review_tool` and `cxr_classification_tool`.
- Added example CXR files under `examples/cxr/`:
  - `torchxrayvision_00000001_000.png`
  - `torchxrayvision_00000001_000_demo.dcm`
- Verified both example PNG and demo DICOM return `status: ok` from `cxr_classification_tool`.
- `cd webapp && npm run build` completed successfully.

Pending local verification:

- Manual browser upload test with `examples/cxr/torchxrayvision_00000001_000.png`.
- Manual browser upload test with `examples/cxr/torchxrayvision_00000001_000_demo.dcm`.
