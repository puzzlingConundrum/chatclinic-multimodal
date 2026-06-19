# cxr_classifier_tool (MedCLIP / BiomedCLIP)

Zero-shot chest X-ray classifier used as the **third model in Stage 2** of `cxr_ensemble_tool`.

## What it does

Runs a vision-language model in zero-shot mode over the 14 CheXpert observation labels
using a prompt ensemble. Backend selection, in order of preference:

1. **BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) via `open_clip` — preferred (biomedical domain).
2. **CLIP** (`openai/clip-vit-base-patch32`) via `transformers` — fallback.

Weights download from HuggingFace on first use. Returns `model_backend`, `top_finding`,
`top_score`, `all_findings`, and `notable_findings`.

## Entry point

```text
plugins.cxr_classifier_tool.logic:classify_cxr   # classify_cxr(image_path=... | dicom_path=...)
```

It is invoked by `cxr_ensemble_tool` during Stage-2 fusion; it can also be called directly.

## Environment / runtime

- Deps in `requirements.txt` (`open_clip_torch`, `transformers`).
- CPU or CUDA; single RTX 3090 is sufficient.
- No checkpoint to submit — weights are fetched from HuggingFace at runtime.

Output is research/screening support only and is not a final clinical diagnosis.
