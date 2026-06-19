from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from math import isclose
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment variables
# ---------------------------------------------------------------------------
LOW_CONF_THRESHOLD = float(os.getenv("CXR_ENSEMBLE_LOW_CONF_THRESHOLD", "0.35"))
MARGIN_THRESHOLD = float(os.getenv("CXR_ENSEMBLE_MARGIN_THRESHOLD", "0.10"))
ENSEMBLE_SCORE_THRESHOLD = float(os.getenv("CXR_ENSEMBLE_SCORE_THRESHOLD", "0.35"))

# Stage-2 escalation decision: "auto" uses the LLM when an OpenAI key is present
# and falls back to the deterministic thresholds otherwise; "llm" forces the LLM
# (still falls back on error); "threshold" forces the legacy hard-coded rule.
DECISION_MODE = os.getenv("CXR_ENSEMBLE_DECISION_MODE", "auto").strip().lower()
DECISION_MODEL = os.getenv("CXR_ENSEMBLE_DECISION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")
DECISION_TIMEOUT = float(os.getenv("CXR_ENSEMBLE_DECISION_TIMEOUT", "30"))
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
# Stage-2 escalation decision — LLM-driven, with deterministic fallback
#
# Instead of hard-coded thresholds, an LLM reads the *raw confidence scores*
# from the Stage-1 DenseNet result and decides whether a single model is
# sufficient or whether to escalate to the 3-model ensemble.  The deterministic
# threshold rule (`_is_low_confidence`) is kept as a transparent fallback for
# offline runs, missing API keys, or API errors, and for the old-vs-new demo.
# ---------------------------------------------------------------------------

def _confidence_summary(txv_result: dict[str, Any]) -> dict[str, Any]:
    """Compact, LLM-friendly view of the Stage-1 raw confidence scores."""
    probs = sorted(
        (txv_result.get("probabilities") or []),
        key=lambda x: float(x.get("score", 0)),
        reverse=True,
    )
    top = probs[0] if probs else {}
    second = probs[1] if len(probs) > 1 else {}
    top_score = float(top.get("score", 0) or 0)
    second_score = float(second.get("score", 0) or 0)
    return {
        "available": bool(txv_result.get("available")),
        "model": str(txv_result.get("model_weights") or DEFAULT_DENSENET_WEIGHTS),
        "score_type": str(txv_result.get("score_type") or "probability"),
        "top_finding": str(top.get("label", "n/a")),
        "top_score": round(top_score, 4),
        "second_finding": str(second.get("label", "n/a")),
        "second_score": round(second_score, 4),
        "margin": round(top_score - second_score, 4),
        "top_k": [
            {"label": str(p.get("label", "n/a")), "score": round(float(p.get("score", 0) or 0), 4)}
            for p in probs[:5]
        ],
    }


_DECISION_SYSTEM_PROMPT = (
    "You are the routing controller for a two-stage chest X-ray (CXR) classification system. "
    "Stage 1 is a single DenseNet-121 CNN that outputs a per-pathology confidence score for each label. "
    "Your only job is to decide whether Stage 1's result can be trusted on its own, or whether to escalate "
    "to Stage 2 — a slower ensemble that adds a ResNet-50 and a vision-language model (MedCLIP) and fuses all "
    "three for a second opinion.\n"
    "Decide from the RAW CONFIDENCE SCORES alone (you are not shown the image):\n"
    "- Escalate when the top score is weak/low (no finding is clearly supported) OR when the margin between the "
    "top and second finding is small (the result is ambiguous between competing findings).\n"
    "- Do NOT escalate when there is a single clearly dominant, high-confidence finding — one model already suffices.\n"
    "These scores are roughly calibrated such that a top score near or below ~0.35 is weak, and a top-1-vs-top-2 "
    "margin under ~0.10 is ambiguous, but use your judgement on the actual numbers rather than a hard cutoff.\n"
    'Respond with ONLY a compact JSON object: {"escalate": <true|false>, "reason": "<one concise sentence that '
    'cites the relevant numbers>"}. No markdown, no extra text.'
)


def _extract_output_text(result: dict[str, Any]) -> str:
    """Pull assistant text out of an OpenAI /v1/responses payload."""
    text = result.get("output_text")
    if text:
        return str(text).strip()
    parts: list[str] = []
    for item in result.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts).strip()


def _parse_decision_json(text: str) -> dict[str, Any] | None:
    """Tolerantly parse the LLM's JSON decision object."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if "escalate" not in obj:
        return None
    return obj


def _llm_escalation_decision(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Ask the LLM to decide on escalation from raw scores. None on any failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    body = {
        "model": DECISION_MODEL,
        "input": [
            {"role": "system", "content": _DECISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Stage-1 DenseNet-121 raw confidence scores:\n"
                    + json.dumps(summary, ensure_ascii=False)
                    + "\n\nShould we escalate to the Stage-2 ensemble?"
                ),
            },
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=DECISION_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    text = _extract_output_text(payload)
    parsed = _parse_decision_json(text)
    if parsed is None:
        return None
    return {
        "escalate": bool(parsed.get("escalate")),
        "reason": str(parsed.get("reason") or "").strip() or "LLM router decision.",
        "model": DECISION_MODEL,
        "raw_text": text,
    }


def _decide_escalation(
    txv_result: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Unified Stage-2 escalation decision. Returns a decision record.

    mode: "llm" | "threshold" | "auto"
    """
    summary = _confidence_summary(txv_result)
    thr_escalate, thr_reason = _is_low_confidence(txv_result)
    threshold_record = {
        "mode": "threshold",
        "escalate": thr_escalate,
        "reason": thr_reason or "Top finding is confident and unambiguous (threshold rule).",
        "raw_confidence": summary,
        "thresholds": {"low_conf": LOW_CONF_THRESHOLD, "margin": MARGIN_THRESHOLD},
    }

    if mode == "threshold":
        return threshold_record

    # Stage-1 unusable → must escalate; LLM has nothing meaningful to judge.
    if not summary["available"] or not summary["top_k"]:
        return {**threshold_record, "mode": "threshold_fallback",
                "fallback_reason": "Stage-1 result unavailable; used threshold rule."}

    llm = _llm_escalation_decision(summary)
    if llm is None:
        return {**threshold_record, "mode": "threshold_fallback",
                "fallback_reason": "LLM decision unavailable (no API key or API error); used threshold rule."}

    return {
        "mode": "llm",
        "escalate": llm["escalate"],
        "reason": llm["reason"],
        "raw_confidence": summary,
        "llm_model": llm["model"],
        "llm_raw_text": llm["raw_text"],
        "threshold_would_be": {"escalate": thr_escalate, "reason": thr_reason},
    }


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
    decision_mode: str = DECISION_MODE,
) -> dict[str, Any]:
    """
    Two-stage CXR classification:

    Stage 1 — Primary (DenseNet121):
        Run DenseNet, then ask the escalation controller whether one model is
        enough.  In the default ("auto"/"llm") mode an LLM reads the raw Stage-1
        confidence scores and decides; in "threshold" mode the legacy rule
        (top score >= low_conf_threshold AND margin >= margin_threshold) decides.

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
            decision_mode=(decision_mode or DECISION_MODE).strip().lower(),
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
    decision_mode: str = DECISION_MODE,
) -> dict[str, Any]:
    base_warnings = [
        "This model output is research/screening support only and is not a final clinical diagnosis.",
        "Use only on chest radiographs; non-CXR inputs may produce misleading results.",
    ]

    # --- Stage 1: DenseNet ---
    densenet_result = _run_txv(source_path, source_kind, densenet_weights)

    # --- Escalation decision (LLM-driven by default, threshold fallback) ---
    decision = _decide_escalation(densenet_result, mode=decision_mode)
    low_conf = bool(decision["escalate"])
    reason = str(decision["reason"])
    decided_by = str(decision["mode"])

    if not low_conf:
        probs = sorted(densenet_result.get("probabilities") or [], key=lambda x: float(x.get("score", 0)), reverse=True)
        top = probs[0] if probs else {}
        return {
            "status": "ok",
            "ensemble_triggered": False,
            "confidence_stage": "primary",
            "low_confidence_reason": None,
            "decision": decision,
            "decided_by": decided_by,
            "stop_reason": reason,
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
                "escalation_decided_by": decided_by,
            },
        }

    # --- Stage 2: Ensemble ---
    ensemble_warnings = base_warnings + [
        f"Low-confidence primary result — escalating to ensemble ({decided_by}). Reason: {reason}",
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
        "decision": decision,
        "decided_by": decided_by,
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
            "escalation_decided_by": decided_by,
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
# Stage-1-only run (for the explicit staged UI: run DenseNet, then let the user
# trigger Stage 2 with an LLM or threshold decision)
# ---------------------------------------------------------------------------

def run_stage1(source_path: str, source_kind: str, densenet_weights: str = DEFAULT_DENSENET_WEIGHTS) -> dict[str, Any]:
    """Run only Stage 1 (DenseNet-121) and report the raw confidence, awaiting Stage 2."""
    base_warnings = [
        "This model output is research/screening support only and is not a final clinical diagnosis.",
        "Use only on chest radiographs; non-CXR inputs may produce misleading results.",
    ]
    densenet_result = _run_txv(source_path, source_kind, densenet_weights)
    summary = _confidence_summary(densenet_result)
    probs = sorted(densenet_result.get("probabilities") or [], key=lambda x: float(x.get("score", 0)), reverse=True)
    top = probs[0] if probs else {}
    return {
        "status": "ok",
        "stage": "stage1",
        "awaiting_stage2": True,
        "ensemble_triggered": False,
        "confidence_stage": "stage1",
        "decided_by": None,
        "low_confidence_reason": None,
        "models_used": [densenet_weights],
        "ensemble_method": "stage1_only",
        "top_finding": str(top.get("label", "n/a")),
        "top_score": float(top.get("score", 0)),
        "aligned_findings": [],
        "positive_findings": [],
        "raw_confidence": summary,
        "model_results": {"densenet121": densenet_result, "resnet50": None, "medclip": None},
        "warnings": base_warnings + list(densenet_result.get("warnings") or []),
        "provenance": {
            "clinical_use": "research_screening_support_only",
            "ensemble_method": "stage1_only",
            "confidence_stage": "stage1",
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

    stage = str(payload.get("stage") or "all").strip().lower()
    densenet_weights = str(payload.get("densenet_weights") or DEFAULT_DENSENET_WEIGHTS)

    if stage in ("1", "stage1"):
        result = run_stage1(str(path), source_kind, densenet_weights)
    else:
        result = run_ensemble(
            str(path),
            source_kind,
            densenet_weights=densenet_weights,
            resnet_weights=str(payload.get("resnet_weights") or DEFAULT_RESNET_WEIGHTS),
            low_conf_threshold=_float("low_conf_threshold", LOW_CONF_THRESHOLD),
            margin_threshold=_float("margin_threshold", MARGIN_THRESHOLD),
            ensemble_threshold=_float("ensemble_threshold", ENSEMBLE_SCORE_THRESHOLD),
            decision_mode=str(payload.get("decision_mode") or DECISION_MODE),
        )
        if stage in ("2", "stage2"):
            result["stage"] = "stage2"

    return {
        "ensemble": result,
        "artifacts": {"cxr_ensemble": result},
        "warnings": result.get("warnings", []),
    }
