# Pull Request Description

## Title

Add confidence-aware two-stage CXR ensemble (LLM-routed) + classification & report-labeling tools

## Summary

This PR adds a chest X-ray diagnosis suite to ChatClinic. The headline contribution is a
**confidence-aware two-stage ensemble** whose escalation decision is made by an **LLM that
reads the Stage-1 raw confidence scores**, replacing hard-coded thresholds. The suite ships
four cooperating plugins plus an explicit staged Studio UI.

### Tools

- `cxr_classification_tool` — single-model TorchXRayVision DenseNet/ResNet presets (Stage 1 / `@cxr`).
- `cxr_classifier_tool` — MedCLIP / BiomedCLIP zero-shot model (used inside Stage 2).
- `cxr_ensemble_tool` — **two-stage orchestrator**: DenseNet-121 → escalation decision → (if escalated) ResNet-50 + MedCLIP with label-aligned weighted score fusion.
- `cxr_report_labeling_tool` — CheXbert/CheXpert-compatible labeling of CXR report text.

### Two-stage routing (the core idea)

1. **Stage 1** runs DenseNet-121 and reports raw confidence (top finding, top score, top-1−top-2 margin).
2. An **escalation controller** decides whether to escalate. `decision_mode=auto` uses an
   **LLM** (`gpt-5-mini`) by default; `threshold` forces the legacy rule; both are kept so
   the old-vs-new behavior can be compared. The LLM keys off the **raw confidence scores**, not the pixels.
3. **Stage 2** (only when escalated) adds ResNet-50 + MedCLIP, aligns the 18/14-label
   vocabularies to a canonical set, and fuses via weighted score fusion (0.40 / 0.35 / 0.25).

### Explicit staged UI

Upload now runs **Stage 1 only**; the Studio card shows the DenseNet result and two buttons —
**Run Stage 2 · LLM decision** and **Run Stage 2 · Threshold rule** — so the user triggers
Stage 2 on demand and can re-run with the other decision mode to compare.

## Main Changes

- New plugins: `plugins/cxr_ensemble_tool/`, `plugins/cxr_classifier_tool/` (+ existing
  `cxr_classification_tool`, `cxr_report_labeling_tool`).
- `plugins/cxr_ensemble_tool/logic.py`: `_decide_escalation` / `_llm_escalation_decision`
  (LLM router over raw scores), `run_stage1`, threshold fallback, `decision_mode` + `stage` params.
- `app/services/workflows.py`: image/DICOM upload runs **Stage 1 only**; Stage 2 runs via the
  tool endpoint.
- Frontend (`webapp/app/...`): CXR Ensemble Studio card with staged states + "Run Stage 2"
  buttons; `handleRunCxrStage2` wired through `page.tsx` and the renderer registry; card shows
  the escalation decision and `decided_by`.
- `skills/chatgenome-orchestrator/SKILL.md`: routing note for the LLM-routed ensemble.
- `requirements.txt` / `environment.yml`: added `open_clip_torch` (Stage-2 MedCLIP/BiomedCLIP).
- Examples: `examples/cxr/demo_cases/` (one confident → single-call image, one ambiguous →
  two-stage image, plus reference CXRs) and `examples/cxr/compare_decision_modes.py`.

## Safety / Clinical Framing

- All outputs are model-derived screening/research support, not a final clinical diagnosis.
- CXR-only; CT/MRI/NIfTI/ultrasound/pathology/non-medical images are out of scope.
- Grounded chat describes probabilities/decisions and must not invent pixel findings.

## Runtime

- Base env: `environment.yml`. Stage-2 adds `open_clip_torch` + `transformers`.
- **No OpenAI key required for inference:** `decision_mode=auto` falls back to the
  deterministic threshold rule, so the full pipeline runs offline.
- CPU or CUDA; verified on a single GPU and designed to run on **up to 4× RTX 3090**.
- Checkpoints: TorchXRayVision DenseNet/ResNet weights in `checkpoints/torchxrayvision/`
  (~226 MB total). MedCLIP/BiomedCLIP weights auto-download from HuggingFace at first use.

## Verification

See `TEST_LOG.md`. Highlights:
- `py_compile` passes for the plugins and `app/services/*.py`; frontend compiles cleanly.
- Stage-1-only upload returns `stage=stage1`, `awaiting_stage2=true`, 1 model.
- Stage-2 via `POST /api/v1/tools/cxr_ensemble/run` returns the full ensemble; `decision_mode`
  `llm` and `threshold` both verified (escalate and single-call paths).
- LLM router routes the two demo images correctly (confident → single call; ambiguous → 3-model ensemble).

## Known Follow-ups

- Export slides to PDF and attach the demo videos for the final submission.
- Quantitative AUC benchmark on a held-out CXR test set; calibrate fusion weights/thresholds.
