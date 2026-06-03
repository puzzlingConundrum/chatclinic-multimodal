from __future__ import annotations

import os
import uuid
from typing import Any

from app.models import (
    AnalysisFacts,
    AnalysisResponse,
    DicomSourceResponse,
    FhirSourceResponse,
    ImageSourceResponse,
    NiftiSourceResponse,
    PrsPrepResponse,
    RawQcResponse,
    SpreadsheetSourceResponse,
    SummaryStatsResponse,
    SymbolicAltSummary,
    TextSourceResponse,
)
from app.services.tool_runner import discover_tools, run_tool
from app.services.workflow_responses import assemble_analysis_response_from_vcf_context
from plugins.fastqc_execution_tool.logic import FASTQC_OUTPUT_DIR
from plugins.cohort_sheet_browser_tool.logic import analyze_spreadsheet_source
from plugins.dicom_review_tool.logic import analyze_dicom_source
from plugins.fhir_browser_tool.logic import analyze_fhir_source
from plugins.image_review_tool.logic import analyze_image_source
from plugins.nifti_review_tool.logic import analyze_nifti_source
from plugins.prs_prep_tool.logic import analyze_prs_prep
from plugins.summary_stats_review_tool.logic import analyze_summary_stats
from plugins.text_review_tool.logic import analyze_text_source


def _attach_cxr_classification_result(
    result: ImageSourceResponse | DicomSourceResponse,
    *,
    source_path: str | None,
    payload_key: str,
) -> ImageSourceResponse | DicomSourceResponse:
    if not source_path:
        return result

    warnings = list(result.warnings or [])
    try:
        classification_payload = run_tool(
            "cxr_classification_tool",
            {
                payload_key: source_path,
                "file_name": result.file_name,
            },
        )
    except Exception as exc:
        classification_result = {
            "available": False,
            "status": "workflow_error",
            "tool": "cxr_classification_tool",
            "source_path": source_path,
            "probabilities": [],
            "top_predictions": [],
            "positive_findings": [],
            "warnings": [f"CXR classification workflow failed: {exc}"],
            "provenance": {"clinical_use": "research_screening_support_only"},
        }
        warnings.extend(classification_result["warnings"])
    else:
        raw_result = classification_payload.get("result")
        classification_result = raw_result if isinstance(raw_result, dict) else classification_payload
        for warning in classification_payload.get("warnings", []):
            warning_text = str(warning).strip()
            if warning_text and warning_text not in warnings:
                warnings.append(warning_text)

    artifacts = dict(result.artifacts or {})
    artifacts["cxr_classification"] = classification_result

    studio_cards = list(result.studio_cards or [])
    if not any(str(card.get("id")) == "cxr_classification" for card in studio_cards if isinstance(card, dict)):
        studio_cards.append(
            {
                "id": "cxr_classification",
                "title": "CXR Classification",
                "subtitle": "TorchXRayVision pathology probabilities",
            }
        )

    used_tools = list(dict.fromkeys([*(result.used_tools or []), "cxr_classification_tool"]))
    return result.model_copy(
        update={
            "artifacts": artifacts,
            "studio_cards": studio_cards,
            "warnings": warnings,
            "used_tools": used_tools,
        }
    )


def _attach_cxr_report_labeling_result(
    result: TextSourceResponse,
    *,
    source_path: str | None,
) -> TextSourceResponse:
    if not source_path:
        return result

    try:
        labeling_payload = run_tool(
            "cxr_report_labeling_tool",
            {
                "text_path": source_path,
                "file_name": result.file_name,
            },
        )
    except Exception:
        return result

    raw_result = labeling_payload.get("result")
    labeling_result = raw_result if isinstance(raw_result, dict) else labeling_payload
    should_attach = bool(labeling_result.get("likely_cxr_report")) or any(
        row.get("value") is not None
        for row in labeling_result.get("labels", [])
        if isinstance(row, dict)
    )
    if not should_attach:
        return result

    warnings = list(result.warnings or [])
    for warning in labeling_payload.get("warnings", []):
        warning_text = str(warning).strip()
        if warning_text and warning_text not in warnings:
            warnings.append(warning_text)

    artifacts = dict(result.artifacts or {})
    artifacts["cxr_report_labels"] = labeling_result

    studio_cards = list(result.studio_cards or [])
    if not any(str(card.get("id")) == "cxr_report_labels" for card in studio_cards if isinstance(card, dict)):
        studio_cards.append(
            {
                "id": "cxr_report_labels",
                "title": "CXR Report Labels",
                "subtitle": "CheXbert/CheXpert-compatible observations",
            }
        )

    used_tools = list(dict.fromkeys([*(result.used_tools or []), "cxr_report_labeling_tool"]))
    return result.model_copy(
        update={
            "artifacts": artifacts,
            "studio_cards": studio_cards,
            "warnings": warnings,
            "used_tools": used_tools,
        }
    )


def _vcf_workflow_context(
    path: str,
    annotation_scope: str,
    annotation_limit: int | None,
) -> dict[str, Any]:
    return {
        "source_vcf_path": path,
        "annotation_scope": annotation_scope,
        "annotation_limit": annotation_limit,
        "max_examples": int(os.getenv("MAX_EXAMPLE_VARIANTS", "8")),
        "used_tools": [],
        "tool_registry": discover_tools(),
        "facts": None,
        "annotations": [],
        "roh_segments": [],
        "snpeff_result": None,
        "candidate_variants": [],
        "clinvar_summary": [],
        "consequence_summary": [],
        "clinical_coverage_summary": [],
        "filtering_summary": [],
        "symbolic_alt_summary": SymbolicAltSummary(count=0, examples=[]),
        "references": [],
        "recommendations": [],
        "ui_cards": [],
        "draft_answer": "",
    }


def analyze_vcf_workflow(
    path: str,
    annotation_scope: str = "representative",
    annotation_limit: int | None = None,
) -> AnalysisResponse:
    max_examples = int(os.getenv("MAX_EXAMPLE_VARIANTS", "8"))
    result = run_tool("vcf_qc_tool", {"vcf_path": path, "max_examples": max_examples})
    facts = AnalysisFacts(**dict(result.get("facts") or {}))
    context = _vcf_workflow_context(path, annotation_scope=annotation_scope, annotation_limit=annotation_limit)
    context["facts"] = facts
    context["used_tools"].append("vcf_qc_tool")
    response = assemble_analysis_response_from_vcf_context(context)
    response.studio = {"renderer": "qc"}
    response.requested_view = "qc"
    return response


def analyze_raw_qc_workflow(path: str, original_name: str) -> RawQcResponse:
    try:
        result = run_tool(
            "fastqc_execution_tool",
            {
                "raw_path": path,
                "original_name": original_name,
            },
        )
        return RawQcResponse(**result)
    except Exception as exc:
        raise RuntimeError(f"Raw QC failed: {exc}") from exc


def analyze_summary_stats_workflow(
    path: str,
    original_name: str,
    genome_build: str = "unknown",
    trait_type: str = "unknown",
) -> SummaryStatsResponse:
    result = analyze_summary_stats(path, original_name, genome_build=genome_build, trait_type=trait_type)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_text_workflow(path: str, original_name: str) -> TextSourceResponse:
    result = analyze_text_source(path, original_name)
    result = _attach_cxr_report_labeling_result(result, source_path=result.source_text_path)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_spreadsheet_workflow(path: str, original_name: str) -> SpreadsheetSourceResponse:
    result = analyze_spreadsheet_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_dicom_workflow(path: str, original_name: str) -> DicomSourceResponse:
    result = analyze_dicom_source(path, original_name)
    result = _attach_cxr_classification_result(
        result,
        source_path=result.source_dicom_path,
        payload_key="dicom_path",
    )
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_fhir_workflow(path: str, original_name: str) -> FhirSourceResponse:
    result = analyze_fhir_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_image_workflow(path: str, original_name: str) -> ImageSourceResponse:
    result = analyze_image_source(path, original_name)
    result = _attach_cxr_classification_result(
        result,
        source_path=result.source_image_path,
        payload_key="image_path",
    )
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_nifti_workflow(path: str, original_name: str) -> NiftiSourceResponse:
    result = analyze_nifti_source(path, original_name)
    result.analysis_id = str(uuid.uuid4())
    result.tool_registry = discover_tools()
    return result


def analyze_prs_prep_workflow(
    path: str,
    original_name: str,
    genome_build: str = "unknown",
) -> PrsPrepResponse:
    result = analyze_prs_prep(path, original_name, genome_build=genome_build)
    result.analysis_id = str(uuid.uuid4())
    return result
