# CXR demo cases — LLM escalation router

Two images chosen to exercise both routing paths of the LLM-driven Stage-2 controller.

| File | DenseNet-121 Stage-1 | LLM decision | Demo path |
|------|----------------------|--------------|-----------|
| `single_call_confident.png` | Infiltration **0.525**, margin **0.445** | **do not escalate** — clearly dominant finding | **Single call** — DenseNet only |
| `two_stage_ambiguous.png`   | Cardiomegaly **0.306**, margin **0.169** | **escalate** — top score weak (< ~0.35) | **Two-stage** — DenseNet + ResNet50 + MedCLIP |

`two_stage_ambiguous.png` is a copy of the bundled TorchXRayVision test image
(`examples/cxr/torchxrayvision_00000001_000.png`).

## How to reproduce the routing
```bash
# from chatclinic-multimodal/
python examples/cxr/compare_decision_modes.py   # old threshold vs new LLM, both images
```
Or per-image, forcing a mode:
```python
from plugins.cxr_ensemble_tool.logic import run_ensemble
run_ensemble("examples/cxr/demo_cases/single_call_confident.png", "image", decision_mode="llm")
run_ensemble("examples/cxr/demo_cases/two_stage_ambiguous.png",   "image", decision_mode="threshold")
```

## Image sources / licensing
Downloaded from Wikimedia Commons for non-commercial academic demonstration. Verify each
file's license/attribution on its Commons page before any redistribution.

- `single_call_confident.png` — Commons "Chest Xray PA 3-8-2010.png" (clean human PA chest radiograph)
- `two_stage_ambiguous.png` — TorchXRayVision test image `00000001_000.png` (Apache-2.0, mlmed/torchxrayvision)

Only these two clean human chest radiographs are kept; earlier non-CXR reference images
(pathology specimens, a CT slice, an anatomical diagram, an animal X-ray) were removed.

These files are for software testing/demonstration only. Model outputs are screening/research
support, not clinical diagnoses.
