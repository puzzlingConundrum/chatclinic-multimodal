from __future__ import annotations

from pathlib import Path
from typing import Any

# CheXpert 14 pathology labels
CXR_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Prompt ensemble for robust zero-shot classification
PROMPT_TEMPLATES = [
    "chest x-ray with {}",
    "chest radiograph showing {}",
    "radiographic evidence of {}",
]

# Probability threshold for "notable" findings: 1.5× uniform prior
_NOTABLE_THRESHOLD = 1.5 / len(CXR_LABELS)

# Singleton model cache (backend → entry dict)
_MODEL_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loading — BiomedCLIP preferred, standard CLIP fallback
# ---------------------------------------------------------------------------

def _load_model(device: str) -> dict[str, Any]:
    """Return a cached model entry, loading on first call."""
    if "entry" in _MODEL_CACHE:
        return _MODEL_CACHE["entry"]

    # Attempt 1: BiomedCLIP via open_clip (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
    try:
        import open_clip  # type: ignore
        model_tag = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, _, preprocess = open_clip.create_model_and_transforms(model_tag)
        tokenizer = open_clip.get_tokenizer(model_tag)
        model = model.eval().to(device)
        entry: dict[str, Any] = {
            "backend": "biomedclip",
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "device": device,
        }
        _MODEL_CACHE["entry"] = entry
        return entry
    except Exception:
        pass

    # Attempt 2: Standard CLIP via HuggingFace transformers
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
    model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_id).eval().to(device)  # type: ignore[arg-type]
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    entry = {
        "backend": "clip",
        "model": clip_model,
        "processor": clip_processor,
        "device": device,
    }
    _MODEL_CACHE["entry"] = entry
    return entry


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _pil_from_dicom(dicom_path: str):
    """Normalize a DICOM file and return a PIL RGB image."""
    from plugins.dicom_review_tool.logic import _normalize_dicom_array
    from PIL import Image  # type: ignore
    raw = Path(dicom_path).read_bytes()
    normalized, message = _normalize_dicom_array(raw)
    if normalized is None:
        raise ValueError(f"DICOM pixel normalization failed: {message}")
    if normalized.ndim == 2:
        return Image.fromarray(normalized, mode="L").convert("RGB")
    return Image.fromarray(normalized).convert("RGB")


def _pil_from_image_path(image_path: str):
    """Open a standard image file and return a PIL RGB image."""
    from PIL import Image  # type: ignore
    return Image.open(image_path).convert("RGB")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_biomedclip(
    entry: dict[str, Any],
    pil_image: Any,
    labels: list[str],
    templates: list[str],
) -> list[dict[str, Any]]:
    """Zero-shot inference using the open_clip BiomedCLIP backend."""
    import torch  # type: ignore
    model = entry["model"]
    preprocess = entry["preprocess"]
    tokenizer = entry["tokenizer"]
    device = entry["device"]

    texts = [tmpl.format(lbl) for lbl in labels for tmpl in templates]
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        txt_feat = model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ txt_feat.T).squeeze(0)

    n_tmpl = len(templates)
    sims = sims.view(len(labels), n_tmpl).mean(dim=1)
    probs = sims.softmax(dim=0).cpu().tolist()
    return [{"label": lbl, "score": float(p)} for lbl, p in zip(labels, probs)]


def _infer_clip(
    entry: dict[str, Any],
    pil_image: Any,
    labels: list[str],
    templates: list[str],
) -> list[dict[str, Any]]:
    """Zero-shot inference using the transformers CLIP backend."""
    import torch  # type: ignore
    model = entry["model"]
    processor = entry["processor"]
    device = entry["device"]

    texts = [tmpl.format(lbl) for lbl in labels for tmpl in templates]
    inputs = processor(text=texts, images=pil_image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        img_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_feat = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sims = (img_feat @ txt_feat.T).squeeze(0)

    n_tmpl = len(templates)
    sims = sims.view(len(labels), n_tmpl).mean(dim=1)
    probs = sims.softmax(dim=0).cpu().tolist()
    return [{"label": lbl, "score": float(p)} for lbl, p in zip(labels, probs)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_cxr(
    image_path: str | None = None,
    dicom_path: str | None = None,
) -> dict[str, Any]:
    """
    Run zero-shot CXR classification.

    Provide exactly one of image_path or dicom_path.  Returns a dict with
    all_findings (sorted by score desc), top_finding, and metadata.
    """
    import torch  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dicom_path:
        pil_image = _pil_from_dicom(dicom_path)
        source_type = "dicom"
        file_path = dicom_path
    elif image_path:
        pil_image = _pil_from_image_path(image_path)
        source_type = "image"
        file_path = image_path
    else:
        raise ValueError("Provide image_path or dicom_path.")

    entry = _load_model(device)
    backend = entry["backend"]

    if backend == "biomedclip":
        findings = _infer_biomedclip(entry, pil_image, CXR_LABELS, PROMPT_TEMPLATES)
    else:
        findings = _infer_clip(entry, pil_image, CXR_LABELS, PROMPT_TEMPLATES)

    findings_sorted = sorted(findings, key=lambda x: x["score"], reverse=True)
    top = findings_sorted[0]
    notable = [f for f in findings_sorted if f["score"] > _NOTABLE_THRESHOLD]

    return {
        "file_path": file_path,
        "source_type": source_type,
        "model_backend": backend,
        "top_finding": top["label"],
        "top_score": top["score"],
        "all_findings": findings_sorted,
        "notable_findings": notable,
        "labels_evaluated": len(CXR_LABELS),
        "prompt_templates_used": len(PROMPT_TEMPLATES),
        "device": device,
    }


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip() or None
    dicom_path = str(payload.get("dicom_path") or "").strip() or None
    if not image_path and not dicom_path:
        raise ValueError("`image_path` or `dicom_path` is required.")
    result = classify_cxr(image_path=image_path, dicom_path=dicom_path)
    return {"classification": result}
