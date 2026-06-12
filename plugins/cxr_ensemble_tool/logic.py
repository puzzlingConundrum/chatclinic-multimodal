from __future__ import annotations

import os
from math import isclose
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
LOW_CONF_THRESHOLD = float(os.getenv("CXR_ENSEMBLE_LOW_CONF_THRESHOLD", "0.35"))
MARGIN_THRESHOLD = float(os.getenv("CXR_ENSEMBLE_MARGIN_THRESHOLD", "0.10"))
ENSEMBLE_SCORE_THRESHOLD = float(os.getenv("CXR_ENSEMBLE_SCORE_THRESHOLD", "0.35"))
DEFAULT_DENSENET_WEIGHTS = "densenet121-res224-all"
DEFAULT_RESNET_WEIGHTS = "resnet50-res512-all"
ENSEMBLE_MODEL_WEIGHTS: dict[str, float] = {
    "densenet121": 0.40,
    "resnet50": 0.35,
    "medclip": 0.25,
}

# ---------------------------------------------------------------------------
# Label alignment — maps each model's raw label → shared canonical label
#
# TorchXRayVision (DenseNet/ResNet) may output up to 18 labels across its
# presets.  MedCLIP uses the 14-class CheXpert label set.  We align both to
# a common canonical vocabulary so votes can be compared.
# ---------------------------------------------------------------------------
_TXV_TO_CANONICAL: dict[str, str | None] = {
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Infiltration": "Lung Opacity",        # approximate: infiltration ≈ opacity
    "Pneumothorax": "Pneumothorax",
    "Edema": "Edema",
    "Emphysema": None,                     # no MedCLIP equivalent
    "Fibrosis": None,
    "Effusion": "Pleural Effusion",
    "Pleural Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pleural_Thickening": "Pleural Other",
    "Cardiomegaly": "Cardiomegaly",
    "Nodule": "Lung Lesion",               # approximate
    "Mass": "Lung Lesion",                 # approximate (max of Nodule+Mass → Lung Lesion)
    "Hernia": None,
    "Fracture": "Fracture",
    "Lung Opacity": "Lung Opacity",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "No Finding": "No Finding",
    "Lung Lesion": "Lung Lesion",
    "Pleural Other": "Pleural Other",
    "Support Devices": "Support Devices",
    "Airspace Opacity": "Lung Opacity",
}

_MEDCLIP_TO_CANONICAL: dict[str, str] = {
    "No Finding": "No Finding",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Cardiomegaly": "Cardiomegaly",
    "Lung Opacity": "Lung Opacity",
    "Lung Lesion": "Lung Lesion",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Pleural Effusion",
    "Pleural Other": "Pleural Other",
    "Fracture": "Fracture",
    "Support Devices": "Support Devices",
}


# ---------------------------------------------------------------------------
# Confidence check
# ---------------------------------------------------------------------------

def _is_low_confidence(txv_result: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_low_conf, reason) for a TorchXRayVision result dict."""
    if not txv_result.get("available"):
        return True, "DenseNet model unavailable or inference failed"

    probs: list[dict[str, Any]] = txv_result.get("probabilities") or []
    if not probs:
        return True, "DenseNet returned no probability outputs"

    sorted_probs = sorted(probs, key=lambda x: float(x.get("score", 0)), reverse=True)
    top_score = float(sorted_probs[0].get("score", 0))
    second_score = float(sorted_probs[1].get("score", 0)) if len(sorted_probs) > 1 else 0.0

    if top_score < LOW_CONF_THRESHOLD:
        return True, (
            f"Top DenseNet score {top_score:.3f} is below the low-confidence threshold {LOW_CONF_THRESHOLD:.3f}"
        )

    margin = top_score - second_score
    if margin < MARGIN_THRESHOLD:
        return True, (
            f"Score margin between top-1 and top-2 ({margin:.3f}) is below the ambiguity margin {MARGIN_THRESHOLD:.3f} "
            f"— top findings are too similar to trust a single model"
        )

    return False, ""


# ---------------------------------------------------------------------------
# Score extraction helpers
# ---------------------------------------------------------------------------

def _txv_canonical_scores(txv_result: dict[str, Any]) -> dict[str, float]:
    """Extract {canonical_label: max_score} from a TXV result."""
    out: dict[str, float] = {}
    for item in (txv_result.get("probabilities") or []):
        raw = str(item.get("raw_label") or item.get("label") or "").strip()
        canonical = _TXV_TO_CANONICAL.get(raw)
        if canonical is None:
            continue
        score = float(item.get("score", 0))
        out[canonical] = max(out.get(canonical, 0.0), score)
    return out


def _medclip_canonical_scores(medclip_result: dict[str, Any]) -> dict[str, float]:
    """Extract {canonical_label: score} from a MedCLIP classify_cxr result."""
    out: dict[str, float] = {}
    for item in (medclip_result.get("all_findings") or []):
        label = str(item.get("label", "")).strip()
        canonical = _MEDCLIP_TO_CANONICAL.get(label)
        if canonical:
            out[canonical] = float(item.get("score", 0))
    return out


# ---------------------------------------------------------------------------
# Deterministic score fusion
# ---------------------------------------------------------------------------

def _normalize_model_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a model's label scores onto a comparable 0..1 simplex."""
    if not scores:
        return {}
    total = sum(max(score, 0.0) for score in scores.values())
    if isclose(total, 0.0):
        width = float(len(scores))
        return {label: 1.0 / width for label in scores}
    return {label: max(score, 0.0) / total for label, score in scores.items()}


def _consensus_strength(
    *,
    supporting_models: int,
    ensemble_score: float,
    positive_threshold: float,
) -> str:
    if supporting_models >= 2 and ensemble_score >= positive_threshold:
        return "high"
    if supporting_models >= 1 or ensemble_score >= (positive_threshold * 0.75):
        return "mixed"
    return "weak"


def _fused_explanation(
    label: str,
    *,
    rank: int,
    ensemble_score: float,
    supporting_model_names: list[str],
    top_contributor: str | None,
    consensus_strength: str,
) -> str:
    rank_clause = (
        f"{label} ranked highest by weighted fusion ({ensemble_score:.3f})"
        if rank == 1
        else f"{label} ranked #{rank} by weighted fusion ({ensemble_score:.3f})"
    )
    if supporting_model_names:
        support_clause = ", ".join(supporting_model_names)
        contributor_clause = f"; strongest weighted contribution came from {top_contributor}" if top_contributor else ""
        return (
            f"{rank_clause}; "
            f"supportive models: {support_clause}{contributor_clause}. Consensus: {consensus_strength}."
        )
    return (
        f"{rank_clause} despite limited per-model threshold support. "
        f"Consensus: {consensus_strength}."
    )


def _weighted_score_fusion(
    model_scores: dict[str, dict[str, float]],
    *,
    model_weights: dict[str, float],
    model_thresholds: dict[str, float],
    positive_threshold: float,
) -> list[dict[str, Any]]:
    """
    Aggregate aligned label scores via deterministic weighted fusion.

    Each model contributes a normalized score distribution across the canonical
    labels it covers. The ensemble score is the weighted sum of those
    normalized scores, using fixed per-model weights renormalized over the
    models that returned usable scores.
    """
    all_labels: set[str] = {lbl for scores in model_scores.values() for lbl in scores}
    normalized_scores = {
        model_name: _normalize_model_scores(scores)
        for model_name, scores in model_scores.items()
    }
    active_model_weights = {
        model_name: model_weights.get(model_name, 0.0)
        for model_name, scores in normalized_scores.items()
        if scores
    }
    total_active_weight = sum(active_model_weights.values()) or 1.0
    effective_model_weights = {
        model_name: weight / total_active_weight
        for model_name, weight in active_model_weights.items()
    }

    findings: list[dict[str, Any]] = []
    for label in sorted(all_labels):
        per_model: dict[str, float | None] = {}
        per_model_normalized: dict[str, float | None] = {}
        weighted_contributions: dict[str, float] = {}
        supporting_model_names: list[str] = []

        for model_name, scores in model_scores.items():
            score = scores.get(label)
            per_model[model_name] = score
            normalized = normalized_scores.get(model_name, {}).get(label)
            per_model_normalized[model_name] = normalized
            contribution = (normalized or 0.0) * effective_model_weights.get(model_name, 0.0)
            weighted_contributions[model_name] = round(contribution, 6)
            if score is not None and score >= model_thresholds.get(model_name, positive_threshold):
                supporting_model_names.append(model_name)

        ensemble_score = sum(weighted_contributions.values())
        supporting_models = len(supporting_model_names)
        positive = ensemble_score >= positive_threshold
        consensus = _consensus_strength(
            supporting_models=supporting_models,
            ensemble_score=ensemble_score,
            positive_threshold=positive_threshold,
        )
        top_contributor = max(
            weighted_contributions.items(),
            key=lambda item: item[1],
            default=(None, 0.0),
        )[0]

        findings.append({
            "label": label,
            "positive": positive,
            "ensemble_score": round(ensemble_score, 6),
            "model_scores": per_model,
            "normalized_model_scores": per_model_normalized,
            "weighted_contributions": weighted_contributions,
            "supporting_models": supporting_models,
            "supporting_model_names": supporting_model_names,
            "consensus_strength": consensus,
            "_top_contributor": top_contributor,
        })

    ranked_findings = sorted(findings, key=lambda x: (-int(x["positive"]), -x["ensemble_score"], x["label"]))
    for idx, finding in enumerate(ranked_findings, start=1):
        finding["rank"] = idx
        finding["explanation"] = _fused_explanation(
            str(finding["label"]),
            rank=idx,
            ensemble_score=float(finding["ensemble_score"]),
            supporting_model_names=list(finding.get("supporting_model_names") or []),
            top_contributor=str(finding["_top_contributor"]) if finding.get("_top_contributor") else None,
            consensus_strength=str(finding.get("consensus_strength") or "n/a"),
        )
        finding.pop("_top_contributor", None)
    return ranked_findings


# ---------------------------------------------------------------------------
# Sub-model runners
# ---------------------------------------------------------------------------

def _run_txv(source_path: str, source_kind: str, weights: str) -> dict[str, Any]:
    """Run TorchXRayVision via the cxr_classification_tool plugin."""
    from plugins.cxr_classification_tool.logic import run as txv_run  # type: ignore
    payload = {f"{source_kind}_path": source_path, "model_weights": weights}
    raw = txv_run(payload)
    result = raw.get("result")
    return result if isinstance(result, dict) else raw


def _run_medclip(source_path: str, source_kind: str) -> dict[str, Any]:
    """Run MedCLIP via the cxr_classifier_tool plugin."""
    from plugins.cxr_classifier_tool.logic import classify_cxr  # type: ignore
    key = "dicom_path" if source_kind == "dicom" else "image_path"
    return classify_cxr(**{key: source_path})


# ---------------------------------------------------------------------------
# Primary ensemble function
# ---------------------------------------------------------------------------

def run_ensemble(
    source_path: str,
    source_kind: str,
    *,
    densenet_weights: str = DEFAULT_DENSENET_WEIGHTS,
    resnet_weights: str = DEFAULT_RESNET_WEIGHTS,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
    ensemble_threshold: float = ENSEMBLE_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    Two-stage CXR classification:

    Stage 1 — Primary (DenseNet121):
        If top score >= low_conf_threshold AND margin >= margin_threshold,
        return the DenseNet result directly (fast path).

    Stage 2 — Ensemble (DenseNet + ResNet50 + MedCLIP):
        Also run ResNet50 and MedCLIP, align labels to a canonical vocabulary,
        and aggregate via deterministic weighted score fusion.
    """
    global LOW_CONF_THRESHOLD, MARGIN_THRESHOLD  # allow per-call overrides
    _prev_low, _prev_margin = LOW_CONF_THRESHOLD, MARGIN_THRESHOLD
    LOW_CONF_THRESHOLD = low_conf_threshold
    MARGIN_THRESHOLD = margin_threshold
    try:
        return _run_ensemble_inner(
            source_path, source_kind,
            densenet_weights=densenet_weights,
            resnet_weights=resnet_weights,
            ensemble_threshold=ensemble_threshold,
        )
    finally:
        LOW_CONF_THRESHOLD = _prev_low
        MARGIN_THRESHOLD = _prev_margin


def _run_ensemble_inner(
    source_path: str,
    source_kind: str,
    *,
    densenet_weights: str,
    resnet_weights: str,
    ensemble_threshold: float,
) -> dict[str, Any]:
    base_warnings = [
        "This model output is research/screening support only and is not a final clinical diagnosis.",
        "Use only on chest radiographs; non-CXR inputs may produce misleading results.",
    ]

    # --- Stage 1: DenseNet ---
    densenet_result = _run_txv(source_path, source_kind, densenet_weights)
    low_conf, reason = _is_low_confidence(densenet_result)

    if not low_conf:
        probs = sorted(densenet_result.get("probabilities") or [], key=lambda x: float(x.get("score", 0)), reverse=True)
        top = probs[0] if probs else {}
        return {
            "status": "ok",
            "ensemble_triggered": False,
            "confidence_stage": "primary",
            "low_confidence_reason": None,
            "models_used": [densenet_weights],
            "ensemble_method": "primary_only",
            "top_finding": str(top.get("label", "n/a")),
            "top_score": float(top.get("score", 0)),
            "aligned_findings": [],
            "positive_findings": [],
            "model_results": {
                "densenet121": densenet_result,
                "resnet50": None,
                "medclip": None,
            },
            "warnings": base_warnings + list(densenet_result.get("warnings") or []),
            "provenance": {
                "clinical_use": "research_screening_support_only",
                "ensemble_method": "primary_only",
                "confidence_stage": "primary",
            },
        }

    # --- Stage 2: Ensemble ---
    ensemble_warnings = base_warnings + [
        f"Low-confidence primary result — escalating to ensemble. Reason: {reason}",
    ]

    resnet_result = _run_txv(source_path, source_kind, resnet_weights)
    medclip_result = _run_medclip(source_path, source_kind)

    densenet_scores = _txv_canonical_scores(densenet_result)
    resnet_scores = _txv_canonical_scores(resnet_result)
    medclip_scores = _medclip_canonical_scores(medclip_result)

    model_scores = {
        "densenet121": densenet_scores,
        "resnet50": resnet_scores,
        "medclip": medclip_scores,
    }
    model_thresholds = {
        "densenet121": 0.50,
        "resnet50": 0.50,
        "medclip": ensemble_threshold,  # CLIP-based, calibrated lower
    }

    aligned_findings = _weighted_score_fusion(
        model_scores,
        model_weights=ENSEMBLE_MODEL_WEIGHTS,
        model_thresholds=model_thresholds,
        positive_threshold=ensemble_threshold,
    )
    positive_findings = [f for f in aligned_findings if f["positive"]]
    top = aligned_findings[0] if aligned_findings else {}

    # Collect extra warnings from sub-models
    for sub in [densenet_result, resnet_result]:
        for w in (sub.get("warnings") or []):
            if str(w) not in ensemble_warnings:
                ensemble_warnings.append(str(w))

    return {
        "status": "ok",
        "ensemble_triggered": True,
        "confidence_stage": "ensemble",
        "low_confidence_reason": reason,
        "models_used": [densenet_weights, resnet_weights, "medclip"],
        "ensemble_method": "weighted_score_fusion_label_aligned",
        "top_finding": str(top.get("label", "n/a")),
        "top_score": float(top.get("ensemble_score", 0)),
        "aligned_findings": aligned_findings,
        "positive_findings": positive_findings,
        "explanation": str(top.get("explanation", "")) if top else "",
        "model_results": {
            "densenet121": densenet_result,
            "resnet50": resnet_result,
            "medclip": medclip_result,
        },
        "warnings": ensemble_warnings,
        "provenance": {
            "clinical_use": "research_screening_support_only",
            "ensemble_method": "weighted_score_fusion_label_aligned",
            "confidence_stage": "ensemble",
            "label_alignment": "TXV→canonical + MedCLIP→canonical",
            "ensemble_model_weights": ENSEMBLE_MODEL_WEIGHTS,
            "thresholds": {
                "low_conf": LOW_CONF_THRESHOLD,
                "margin": MARGIN_THRESHOLD,
                "ensemble": ensemble_threshold,
                "densenet121_positive": 0.50,
                "resnet50_positive": 0.50,
                "medclip_positive": ensemble_threshold,
            },
        },
    }


# ---------------------------------------------------------------------------
# Plugin entrypoint
# ---------------------------------------------------------------------------

def execute(payload: dict[str, Any]) -> dict[str, Any]:
    image_path = str(payload.get("image_path") or "").strip() or None
    dicom_path = str(payload.get("dicom_path") or "").strip() or None
    if not image_path and not dicom_path:
        raise ValueError("`image_path` or `dicom_path` is required.")

    source_path = dicom_path or image_path
    source_kind = "dicom" if dicom_path else "image"
    path = Path(source_path).expanduser().resolve()  # type: ignore[arg-type]
    if not path.exists():
        raise FileNotFoundError(f"CXR source not found: {path}")

    def _float(key: str, default: float) -> float:
        try:
            return float(str(payload.get(key) or default))
        except (TypeError, ValueError):
            return default

    result = run_ensemble(
        str(path),
        source_kind,
        densenet_weights=str(payload.get("densenet_weights") or DEFAULT_DENSENET_WEIGHTS),
        resnet_weights=str(payload.get("resnet_weights") or DEFAULT_RESNET_WEIGHTS),
        low_conf_threshold=_float("low_conf_threshold", LOW_CONF_THRESHOLD),
        margin_threshold=_float("margin_threshold", MARGIN_THRESHOLD),
        ensemble_threshold=_float("ensemble_threshold", ENSEMBLE_SCORE_THRESHOLD),
    )
    return {
        "ensemble": result,
        "artifacts": {"cxr_ensemble": result},
        "warnings": result.get("warnings", []),
    }
