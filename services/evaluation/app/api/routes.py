"""
Evaluation Service API routes (`/v1/evaluate`).
Exposes evaluation benchmark execution and report retrieval.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse

from services.evaluation.app.models.eval import EvaluationSummaryReport
from services.evaluation.app.services.report_generator import REPORTS_DIR, EvaluationRunnerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/evaluate", tags=["evaluation"])

runner_service = EvaluationRunnerService()


@router.post(
    "/run",
    response_model=EvaluationSummaryReport,
    status_code=status.HTTP_200_OK,
    summary="Trigger evaluation run across baseline and self-correcting RAG",
)
def run_evaluation_benchmark(tenant_id: str = "eval_tenant") -> EvaluationSummaryReport:
    """Runs all 15 questions across both pipelines and exports reports."""
    try:
        report = runner_service.run_evaluation(tenant_id=tenant_id)
        return report
    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation execution failed: {exc}",
        ) from exc


@router.get(
    "/report",
    summary="Retrieve formatted evaluation report (HTML or Markdown)",
)
def get_evaluation_report(format: str = "markdown") -> Any:
    """Returns the latest evaluation report from `reports/`."""
    if format.lower() == "json":
        json_path = REPORTS_DIR / "evaluation_comparison_report.json"
        if not json_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No JSON evaluation report found. Run `/v1/evaluate/run` first.",
            )
        with open(json_path, encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    md_path = REPORTS_DIR / "evaluation_comparison_report.md"
    if not md_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Markdown evaluation report found. Run `/v1/evaluate/run` first.",
        )

    with open(md_path, encoding="utf-8") as f:
        md_content = f.read()

    if format.lower() == "html":
        # Render basic HTML table structure
        import jinja2

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compass RAG Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <pre>{{ content }}</pre>
        </body>
        </html>
        """
        template = jinja2.Template(html_template)
        return HTMLResponse(content=template.render(content=md_content))

    return md_content
