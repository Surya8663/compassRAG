"""
Debug script to run evaluation on 1 question with full exception logging.
"""

import logging
import sys

from services.evaluation.app.services.report_generator import (
    EvaluationRunnerService,
    load_golden_dataset,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("debug_eval")

if __name__ == "__main__":
    try:
        runner = EvaluationRunnerService()
        subset = load_golden_dataset()[:1]
        logger.info("Running evaluation on single question subset: %s", subset[0].id)
        report = runner.run_evaluation(dataset=subset, tenant_id="debug_tenant")
        logger.info("Successfully finished subset run: %s", report.run_id)
    except Exception as exc:
        logger.error("Error encountered during single evaluation: %s", exc, exc_info=True)
        sys.exit(1)
