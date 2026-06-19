# cxr_ensemble_tool

Two-stage, confidence-aware chest X-ray ensemble classifier — the headline tool of this submission.

## What it does

**Stage 1 (always):** runs TorchXRayVision **DenseNet-121** and reports per-pathology
probabilities + a raw-confidence summary (top finding, top score, margin, top-5).

**Escalation decision:** an escalation controller reads the *raw Stage-1 confidence
scores* and decides whether one model is enough or whether to escalate. By default an
**LLM** (`gpt-5-mini` via the OpenAI Responses API) makes this decision, replacing the
old hard-coded thresholds; a deterministic threshold rule is the transparent fallback.

**Stage 2 (only when escalated):** also runs TorchXRayVision **ResNet-50** and a
**MedCLIP / BiomedCLIP** zero-shot classifier, aligns all three label vocabularies to a
canonical set, and fuses them via deterministic weighted score fusion (0.40 / 0.35 / 0.25).

## Entry point

```text
plugins.cxr_ensemble_tool.logic:execute
```

`run_stage1(...)` and `run_ensemble(...)` are also importable for scripted use.

## Key payload options

| Key | Default | Meaning |
|-----|---------|---------|
| `image_path` / `dicom_path` | — | input CXR (one required) |
| `stage` | `all` | `1` = DenseNet only (staged UI); `2` = decision + ensemble; `all` = full run |
| `decision_mode` | `auto` | `auto` (LLM if `OPENAI_API_KEY` set, else threshold), `llm`, or `threshold` |
| `densenet_weights` | `densenet121-res224-all` | Stage-1 weights |
| `resnet_weights` | `resnet50-res512-all` | Stage-2 weights |
| `low_conf_threshold` / `margin_threshold` | 0.35 / 0.10 | thresholds (threshold mode only) |

## Output (selected fields)

```text
ensemble_triggered, decided_by ("llm"|"threshold"|"threshold_fallback"),
decision { mode, escalate, reason, raw_confidence, llm_model, threshold_would_be },
confidence_stage, top_finding, top_score, aligned_findings[], positive_findings[],
model_results { densenet121, resnet50, medclip }, warnings[], provenance{}
```

## Environment / runtime

- Base: repository `environment.yml`. Additional Stage-2 deps in `requirements.txt`
  (`open_clip_torch`, `transformers`).
- DenseNet/ResNet weights load from `checkpoints/torchxrayvision/` (or TorchXRayVision's
  cache); MedCLIP/BiomedCLIP weights auto-download from HuggingFace on first use.
- **No OpenAI key required to run inference:** `decision_mode=auto` falls back to the
  deterministic threshold rule, so the full pipeline runs offline on GPU/CPU.
- CPU or CUDA; runs comfortably on a single RTX 3090 (≤4 supported).

## Smoke test

```bash
python -c "from plugins.cxr_ensemble_tool.logic import run_ensemble; \
print(run_ensemble('examples/cxr/demo_cases/two_stage_ambiguous.png','image',decision_mode='threshold')['top_finding'])"
```

Output is research/screening support only and is not a final clinical diagnosis.
