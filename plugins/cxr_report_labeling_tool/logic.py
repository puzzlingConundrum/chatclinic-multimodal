from __future__ import annotations

import re
from pathlib import Path
from typing import Any


LABELS = [
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

FINDING_PATTERNS: dict[str, list[str]] = {
    "Enlarged Cardiomediastinum": [
        r"enlarged cardiomediastinal silhouette",
        r"widened mediastinum",
        r"mediastinal widening",
    ],
    "Cardiomegaly": [
        r"cardiomegaly",
        r"enlarged cardiac silhouette",
        r"cardiac silhouette is enlarged",
        r"heart is enlarged",
    ],
    "Lung Opacity": [
        r"lung opacity",
        r"pulmonary opacity",
        r"airspace opacity",
        r"airspace disease",
        r"opacification",
        r"infiltrate",
    ],
    "Lung Lesion": [
        r"lung lesion",
        r"pulmonary lesion",
        r"pulmonary nodule",
        r"lung nodule",
        r"pulmonary mass",
        r"lung mass",
    ],
    "Edema": [
        r"pulmonary edema",
        r"interstitial edema",
        r"vascular congestion",
        r"cephalization",
    ],
    "Consolidation": [
        r"consolidation",
        r"consolidative opacity",
        r"airspace consolidation",
    ],
    "Pneumonia": [
        r"pneumonia",
        r"infectious infiltrate",
        r"infectious opacity",
    ],
    "Atelectasis": [
        r"atelectasis",
        r"subsegmental atelectatic",
        r"bibasilar atelectatic",
    ],
    "Pneumothorax": [
        r"pneumothorax",
        r"apical pleural line",
    ],
    "Pleural Effusion": [
        r"pleural effusion",
        r"effusion",
        r"costophrenic angle blunting",
    ],
    "Pleural Other": [
        r"pleural thickening",
        r"pleural plaque",
        r"pleural scarring",
    ],
    "Fracture": [
        r"fracture",
        r"rib deformity",
        r"osseous deformity",
    ],
    "Support Devices": [
        r"endotracheal tube",
        r"\bett\b",
        r"enteric tube",
        r"nasogastric tube",
        r"\bng tube\b",
        r"central venous catheter",
        r"\bcvc\b",
        r"\bpicc\b",
        r"pacemaker",
        r"chest tube",
        r"port catheter",
    ],
}

NO_FINDING_PATTERNS = [
    r"no acute cardiopulmonary abnormality",
    r"no acute cardiopulmonary disease",
    r"no acute disease",
    r"no acute chest abnormality",
    r"clear lungs",
    r"lungs are clear",
    r"unremarkable chest",
]

CHEST_REPORT_CUES = [
    r"\bchest\b",
    r"\bcxr\b",
    r"\bradiograph\b",
    r"\bx-?ray\b",
    r"\bportable ap\b",
    r"\bpa and lateral\b",
    r"\bcardiomediastinal\b",
    r"\bpleural\b",
    r"\bpneumothorax\b",
    r"\bimpression\b",
    r"\bfindings\b",
]

NEGATION_CUES = [
    "no",
    "without",
    "negative for",
    "no evidence of",
    "no focal",
    "no definite",
    "no pleural",
    "no pneumothorax",
    "no acute",
    "absent",
]

UNCERTAIN_CUES = [
    "possible",
    "possibly",
    "probable",
    "probably",
    "may represent",
    "may reflect",
    "cannot exclude",
    "difficult to exclude",
    "questionable",
    "suspected",
    "suggests",
    "suggestive",
]


def _read_report_text(payload: dict[str, object]) -> tuple[str, str | None]:
    inline = str(payload.get("report_text") or "").strip()
    if inline:
        return inline, None
    text_path = str(payload.get("text_path") or "").strip()
    if not text_path:
        raise ValueError("`text_path` or `report_text` is required.")
    path = Path(text_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CXR report text not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace"), str(path)


def _sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|(?:\n\s*){2,}", collapsed)
    return [piece.strip() for piece in pieces if piece.strip()]


def _contains_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _match_patterns(sentence: str, patterns: list[str]) -> list[str]:
    lowered = sentence.lower()
    matches: list[str] = []
    for pattern in patterns:
        if re.search(pattern, lowered):
            matches.append(pattern)
    return matches


def _cue_window(sentence: str, pattern: str, window_chars: int = 70) -> str:
    lowered = sentence.lower()
    match = re.search(pattern, lowered)
    if not match:
        return lowered
    start = max(0, match.start() - window_chars)
    end = min(len(lowered), match.end() + window_chars)
    return lowered[start:end]


def _classify_evidence(sentence: str, pattern: str) -> tuple[int, str]:
    window = _cue_window(sentence, pattern)
    if any(cue in window for cue in UNCERTAIN_CUES):
        return -1, "uncertain"
    if any(cue in window for cue in NEGATION_CUES):
        return 0, "negative"
    return 1, "positive"


def _label_class(value: int | None) -> str:
    if value == 1:
        return "positive"
    if value == 0:
        return "negative"
    if value == -1:
        return "uncertain"
    return "blank"


def _aggregate(values: list[int]) -> int | None:
    if 1 in values:
        return 1
    if -1 in values:
        return -1
    if 0 in values:
        return 0
    return None


def _is_likely_cxr_report(text: str) -> bool:
    return _contains_any(text, CHEST_REPORT_CUES)


def _label_report(text: str) -> list[dict[str, Any]]:
    sentences = _sentences(text)
    label_rows: list[dict[str, Any]] = []

    for label in LABELS:
        evidence: list[dict[str, Any]] = []
        values: list[int] = []

        if label == "No Finding":
            for sentence in sentences:
                for pattern in _match_patterns(sentence, NO_FINDING_PATTERNS):
                    evidence.append(
                        {
                            "sentence": sentence,
                            "matched_pattern": pattern,
                            "polarity": "positive",
                        }
                    )
                    values.append(1)
            label_rows.append(
                {
                    "label": label,
                    "value": _aggregate(values),
                    "label_class": _label_class(_aggregate(values)),
                    "evidence": evidence[:4],
                }
            )
            continue

        for sentence in sentences:
            for pattern in _match_patterns(sentence, FINDING_PATTERNS.get(label, [])):
                value, polarity = _classify_evidence(sentence, pattern)
                evidence.append(
                    {
                        "sentence": sentence,
                        "matched_pattern": pattern,
                        "polarity": polarity,
                    }
                )
                values.append(value)

        value = _aggregate(values)
        label_rows.append(
            {
                "label": label,
                "value": value,
                "label_class": _label_class(value),
                "evidence": evidence[:4],
            }
        )

    non_no_finding_positive = any(row["label"] != "No Finding" and row["value"] == 1 for row in label_rows)
    if non_no_finding_positive:
        for row in label_rows:
            if row["label"] == "No Finding" and row["value"] == 1:
                row["value"] = 0
                row["label_class"] = "negative"
                row["evidence"].append(
                    {
                        "sentence": "Positive report observations were found, so No Finding is not treated as positive.",
                        "matched_pattern": "aggregation_override",
                        "polarity": "negative",
                    }
                )
                break

    return label_rows


def run(payload: dict[str, object]) -> dict[str, object]:
    text, source_path = _read_report_text(payload)
    labels = _label_report(text)
    likely_report = _is_likely_cxr_report(text)

    positive_labels = [row for row in labels if row["value"] == 1]
    negative_labels = [row for row in labels if row["value"] == 0]
    uncertain_labels = [row for row in labels if row["value"] == -1]
    blank_labels = [row for row in labels if row["value"] is None]

    warnings = [
        "Labels are derived from report text only and are not new image findings.",
        "This lightweight backend follows the CheXpert label convention but is not the full CheXbert BERT checkpoint.",
    ]
    if not likely_report:
        warnings.append("The text does not strongly look like a chest radiology report; labels may be uninformative.")

    top_summary = ", ".join(row["label"] for row in positive_labels[:4]) or "no positive observation labels"
    result = {
        "available": True,
        "status": "ok",
        "tool": "cxr_report_labeling_tool",
        "source_path": source_path,
        "model_family": "CheXbert/CheXpert-compatible report labeler",
        "backend": "rules",
        "label_convention": {
            "positive": 1,
            "negative": 0,
            "uncertain": -1,
            "blank": None,
        },
        "likely_cxr_report": likely_report,
        "report_character_count": len(text),
        "labels": labels,
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
        "uncertain_labels": uncertain_labels,
        "blank_labels": blank_labels,
        "warnings": warnings,
        "provenance": {
            "primary_reference": "CheXbert",
            "label_set": "CheXpert 14 observations",
            "implementation": "deterministic local rule backend",
            "clinical_use": "report_text_screening_support_only",
            "sources": [
                "https://github.com/stanfordmlgroup/CheXbert",
                "https://github.com/stanfordmlgroup/chexpert-labeler",
                "https://arxiv.org/abs/2004.09167",
                "https://arxiv.org/abs/1901.07031",
            ],
        },
    }
    summary = f"CXR report labeling completed. Positive labels: {top_summary}."
    return {
        "tool": "cxr_report_labeling_tool",
        "summary": summary,
        "result": result,
        "artifacts": {"cxr_report_labels": result},
        "warnings": warnings,
        "provenance": result["provenance"],
    }


def execute(payload: dict[str, object]) -> dict[str, object]:
    return run(payload)
