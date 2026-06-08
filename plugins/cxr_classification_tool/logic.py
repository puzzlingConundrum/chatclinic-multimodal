from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT_DIR / "checkpoints" / "torchxrayvision"
DEFAULT_WEIGHTS = "densenet121-res224-all"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
DEFAULT_RESOLUTION = 224
DEFAULT_USE_SOFTMAX = True
DEFAULT_SOFTMAX_C = 1.5  # threshold = C / num_valid_classes
MODEL_PRESETS: dict[str, dict[str, object]] = {
    "densenet121-res224-all": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "TorchXRayVision all-data blend",
        "description": "General multi-dataset CXR pathology model; all 18 outputs are trained.",
    },
    "densenet121-res224-rsna": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "RSNA Pneumonia Detection Challenge",
        "description": "Pneumonia-focused RSNA pretrained CXR model.",
    },
    "densenet121-res224-nih": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "NIH ChestX-ray14",
        "description": "NIH ChestX-ray14 pretrained CXR pathology model.",
    },
    "densenet121-res224-pc": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "PadChest",
        "description": "PadChest pretrained CXR pathology model.",
    },
    "densenet121-res224-chex": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "CheXpert",
        "description": "CheXpert pretrained CXR pathology model.",
    },
    "densenet121-res224-mimic_nb": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "MIMIC-CXR, NIH-style label mapping",
        "description": "MIMIC-CXR pretrained model using the NIH label basis.",
    },
    "densenet121-res224-mimic_ch": {
        "architecture": "DenseNet121",
        "resolution": 224,
        "trained_on": "MIMIC-CXR, CheXpert-style label mapping",
        "description": "MIMIC-CXR pretrained model using the CheXpert label basis.",
    },
    "resnet50-res512-all": {
        "architecture": "ResNet50",
        "resolution": 512,
        "trained_on": "TorchXRayVision all-data blend",
        "description": "Higher-resolution 512px TorchXRayVision ResNet model.",
    },
}
MODEL_ALIASES = {
    "all": "densenet121-res224-all",
    "rsna": "densenet121-res224-rsna",
    "nih": "densenet121-res224-nih",
    "pc": "densenet121-res224-pc",
    "padchest": "densenet121-res224-pc",
    "chex": "densenet121-res224-chex",
    "chexpert": "densenet121-res224-chex",
    "mimic_nb": "densenet121-res224-mimic_nb",
    "mimic_ch": "densenet121-res224-mimic_ch",
    "resnet": "resnet50-res512-all",
    "resnet50": "resnet50-res512-all",
}


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cache_dir_from_env() -> str | None:
    explicit = os.getenv("TORCHXRAYVISION_CACHE_DIR", "").strip()
    if explicit:
        return explicit
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(DEFAULT_CACHE_DIR)


def _missing_dependency_result(path: Path, source_kind: str, reason: str) -> dict[str, Any]:
    warning = (
        "CXR classification could not run because optional TorchXRayVision runtime dependencies "
        f"are not available: {reason}"
    )
    result = {
        "available": False,
        "status": "dependency_missing",
        "tool": "cxr_classification_tool",
        "source_path": str(path),
        "source_kind": source_kind,
        "model_family": "TorchXRayVision",
        "model_weights": DEFAULT_WEIGHTS,
        "model_preset": MODEL_PRESETS[DEFAULT_WEIGHTS],
        "available_model_weights": sorted(MODEL_PRESETS),
        "probabilities": [],
        "top_predictions": [],
        "positive_findings": [],
        "threshold": DEFAULT_THRESHOLD,
        "device": "not-run",
        "warnings": [warning],
        "provenance": {
            "library": "torchxrayvision",
            "library_available": False,
            "clinical_use": "research_screening_support_only",
        },
    }
    return {
        "tool": "cxr_classification_tool",
        "summary": warning,
        "result": result,
        "artifacts": {"cxr_classification": result},
        "warnings": [warning],
        "provenance": result["provenance"],
    }


def _model_unavailable_result(
    path: Path,
    source_kind: str,
    model_weights: str,
    threshold: float,
    reason: str,
) -> dict[str, Any]:
    warning = f"CXR classification model could not be loaded or executed: {reason}"
    result = {
        "available": False,
        "status": "model_unavailable",
        "tool": "cxr_classification_tool",
        "source_path": str(path),
        "source_kind": source_kind,
        "model_family": "TorchXRayVision",
        "model_weights": model_weights,
        "model_preset": MODEL_PRESETS.get(model_weights),
        "available_model_weights": sorted(MODEL_PRESETS),
        "probabilities": [],
        "top_predictions": [],
        "positive_findings": [],
        "threshold": threshold,
        "device": "not-run",
        "warnings": [warning],
        "provenance": {
            "library": "torchxrayvision",
            "library_available": True,
            "cache_dir": _cache_dir_from_env() or "~/.torchxrayvision",
            "clinical_use": "research_screening_support_only",
        },
    }
    return {
        "tool": "cxr_classification_tool",
        "summary": warning,
        "result": result,
        "artifacts": {"cxr_classification": result},
        "warnings": [warning],
        "provenance": result["provenance"],
    }


def _resolve_source_path(payload: dict[str, object]) -> tuple[Path, str]:
    image_path = str(payload.get("image_path") or "").strip()
    dicom_path = str(payload.get("dicom_path") or "").strip()
    if image_path:
        return Path(image_path).expanduser().resolve(), "image"
    if dicom_path:
        return Path(dicom_path).expanduser().resolve(), "dicom"
    raise ValueError("`image_path` or `dicom_path` is required.")


def _resolve_device(torch: Any, requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized and normalized not in {"auto", "gpu"}:
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return normalized
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_weights(requested: object) -> str:
    normalized = str(requested or DEFAULT_WEIGHTS).strip()
    if not normalized:
        return DEFAULT_WEIGHTS
    normalized = MODEL_ALIASES.get(normalized.lower(), normalized)
    if normalized not in MODEL_PRESETS:
        valid = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unsupported CXR model_weights `{normalized}`. Valid options: {valid}")
    return normalized


def _load_xray_array(xrv: Any, path: Path, source_kind: str) -> Any:
    if source_kind == "dicom" and hasattr(xrv.utils, "read_xray_dcm"):
        try:
            image = xrv.utils.read_xray_dcm(str(path), voi_lut=False, fix_monochrome=True)
        except TypeError:
            image = xrv.utils.read_xray_dcm(str(path))
    else:
        image = xrv.utils.load_image(str(path))
    if getattr(image, "ndim", 0) == 2:
        image = image[None, ...]
    if getattr(image, "ndim", 0) == 3 and image.shape[0] not in (1, 3):
        image = image.transpose(2, 0, 1)
    if getattr(image, "ndim", 0) == 3 and image.shape[0] == 3:
        image = image.mean(axis=0, keepdims=True)
    return image


def _load_model(xrv: Any, model_weights: str, cache_dir: str | None) -> Any:
    # Load with apply_sigmoid=False.  Models that ship with op_threshs (e.g. the
    # -all blended models) apply sigmoid internally in their forward pass and then
    # run op_norm calibration.  Passing apply_sigmoid=True on those models causes
    # a double-sigmoid (sigmoid → sigmoid → op_norm) that collapses all scores to
    # ~0.75 regardless of the input.  We apply sigmoid manually below for models
    # that have no op_threshs.
    model_class = xrv.models.ResNet if model_weights.startswith("resnet") else xrv.models.DenseNet
    try:
        return model_class(weights=model_weights, apply_sigmoid=False, cache_dir=cache_dir)
    except TypeError:
        return model_class(weights=model_weights, apply_sigmoid=False)


def run(payload: dict[str, object]) -> dict[str, object]:
    path, source_kind = _resolve_source_path(payload)
    if not path.exists():
        raise FileNotFoundError(f"CXR source not found: {path}")

    requested_model_weights = str(payload.get("model_weights") or DEFAULT_WEIGHTS).strip() or DEFAULT_WEIGHTS
    threshold = _as_float(payload.get("threshold"), DEFAULT_THRESHOLD)
    use_softmax = str(payload.get("use_softmax", DEFAULT_USE_SOFTMAX)).lower() not in ("false", "0", "no")
    softmax_c = _as_float(payload.get("softmax_c"), DEFAULT_SOFTMAX_C)
    try:
        model_weights = _resolve_model_weights(requested_model_weights)
    except ValueError as exc:
        return _model_unavailable_result(path, source_kind, requested_model_weights, threshold, str(exc))
    model_preset = MODEL_PRESETS[model_weights]
    top_k = max(1, int(_as_float(payload.get("top_k"), DEFAULT_TOP_K)))
    default_resolution = int(model_preset.get("resolution") or DEFAULT_RESOLUTION)
    resolution = max(128, int(_as_float(payload.get("resolution"), default_resolution)))
    requested_device = str(payload.get("device") or "auto").strip()

    try:
        import torch
        import torchvision
        import torchxrayvision as xrv
    except Exception as exc:
        return _missing_dependency_result(path, source_kind, str(exc))

    cache_dir = _cache_dir_from_env()
    try:
        device = _resolve_device(torch, requested_device)
        transform = torchvision.transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(resolution),
            ]
        )
        image = _load_xray_array(xrv, path, source_kind)
        image = transform(image)
        image_tensor = torch.from_numpy(image).float()[None, ...].to(device)

        model = _load_model(xrv, model_weights, cache_dir)
        model = model.to(device)
        model.eval()

        labels = list(getattr(model, "pathologies", None) or getattr(model, "targets", None) or [])
        valid_indices = [i for i, lbl in enumerate(labels) if str(lbl or "").strip()]
        n_valid = len(valid_indices) or 1  # guard against empty

        if use_softmax:
            # Softmax over raw logits: represents relative probability distribution.
            # Threshold = C / N (positive if label's share is C× above uniform).
            # Temporarily bypass internal sigmoid/op_norm to get raw logits.
            saved_threshs = getattr(model, "op_threshs", None)
            if saved_threshs is not None:
                model.op_threshs = None
            try:
                with torch.no_grad():
                    raw_logits = model(image_tensor)
            finally:
                if saved_threshs is not None:
                    model.op_threshs = saved_threshs

            valid_logits = raw_logits[0, valid_indices]
            softmax_probs = torch.softmax(valid_logits, dim=0).cpu().numpy()
            score_map = {vi: float(softmax_probs[i]) for i, vi in enumerate(valid_indices)}
            scores = [score_map.get(i) for i in range(len(labels))]
            threshold_used = softmax_c / n_valid
            score_method = "softmax"
        else:
            # Sigmoid / op_norm calibrated path.
            with torch.no_grad():
                output = model(image_tensor)
                if not (hasattr(model, "op_threshs") and model.op_threshs is not None):
                    output = torch.sigmoid(output)
            scores = [s if str(labels[i] or "").strip() else None
                      for i, s in enumerate(output.cpu().numpy()[0].tolist())]
            threshold_used = threshold
            score_method = "sigmoid_op_norm"

        if not labels:
            labels = [f"class_{i}" for i in range(len(scores))]

        probabilities = []
        omitted_untrained_outputs = 0
        for index, score in enumerate(scores):
            raw_label = str(labels[index] if index < len(labels) else "").strip()
            if not raw_label:
                omitted_untrained_outputs += 1
                continue
            if score is None:
                continue
            probabilities.append(
                {
                    "label": raw_label.replace("_", " "),
                    "raw_label": raw_label,
                    "score": round(float(score), 6),
                    "positive": float(score) >= threshold_used,
                }
            )
        sorted_predictions = sorted(probabilities, key=lambda item: float(item["score"]), reverse=True)
        top_predictions = sorted_predictions[:top_k]
        positive_findings = [item for item in sorted_predictions if bool(item["positive"])]
        top_label = top_predictions[0]["label"] if top_predictions else "not available"
        top_score = top_predictions[0]["score"] if top_predictions else "n/a"

        warnings = [
            "This model output is research/screening support only and is not a final clinical diagnosis.",
            "Use only for chest radiographs; non-CXR inputs may produce misleading probabilities.",
        ]
        if use_softmax:
            warnings.append(
                "Softmax scores show relative probability across all classes (sum = 1). "
                "Co-occurring pathologies may appear lower than their true individual probability."
            )
        if omitted_untrained_outputs:
            warnings.append(
                f"{omitted_untrained_outputs} raw TorchXRayVision outputs were omitted "
                "because this preset does not define trained labels for them."
            )
        result = {
            "available": True,
            "status": "ok",
            "tool": "cxr_classification_tool",
            "source_path": str(path),
            "source_kind": source_kind,
            "model_family": "TorchXRayVision",
            "model_weights": model_weights,
            "model_preset": model_preset,
            "available_model_weights": sorted(MODEL_PRESETS),
            "input_resolution": resolution,
            "score_method": score_method,
            "softmax_c": softmax_c if use_softmax else None,
            "num_valid_classes": n_valid,
            "threshold": round(threshold_used, 6),
            "device": device,
            "raw_output_count": len([s for s in scores if s is not None]),
            "omitted_untrained_output_count": omitted_untrained_outputs,
            "probabilities": probabilities,
            "top_predictions": top_predictions,
            "positive_findings": positive_findings,
            "warnings": warnings,
            "provenance": {
                "library": "torchxrayvision",
                "library_available": True,
                "architecture": model_preset.get("architecture"),
                "trained_on": model_preset.get("trained_on"),
                "cache_dir": cache_dir or "~/.torchxrayvision",
                "preprocessing": [
                    "TorchXRayVision load_image/read_xray_dcm",
                    "XRayCenterCrop",
                    f"XRayResizer({resolution})",
                    f"softmax(C={softmax_c}, threshold={round(threshold_used, 4)})" if use_softmax else "sigmoid + op_norm",
                ],
                "clinical_use": "research_screening_support_only",
            },
        }
        summary = (
            f"CXR classification completed with `{model_weights}`. "
            f"Top model finding: {top_label} ({top_score:.3f})."
            if isinstance(top_score, float) else
            f"CXR classification completed with `{model_weights}`. "
            f"Top model finding: {top_label} ({top_score})."
        )
        return {
            "tool": "cxr_classification_tool",
            "summary": summary,
            "result": result,
            "artifacts": {"cxr_classification": result},
            "warnings": warnings,
            "provenance": result["provenance"],
        }
    except Exception as exc:
        return _model_unavailable_result(path, source_kind, model_weights, threshold, str(exc))


def execute(payload: dict[str, object]) -> dict[str, object]:
    return run(payload)
