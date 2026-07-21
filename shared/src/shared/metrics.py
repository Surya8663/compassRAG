"""
Shared Prometheus metrics definitions for Compass RAG services.

All metric objects are module-level singletons created once at import time.
Each service calls `setup_metrics(app)` at startup to expose the /metrics endpoint.

Metric catalogue:
- compass_http_requests_total          Counter   service, method, path, status_code
- compass_http_request_latency_seconds Histogram service, path
- compass_correction_loop_retries_total Counter  service, tenant_id
- compass_pipeline_stage_duration_seconds Histogram service, stage
- compass_active_requests              Gauge     service
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions (module-level singletons)
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    name="compass_http_requests_total",
    documentation="Total HTTP requests handled by each Compass RAG service.",
    labelnames=["service", "method", "path", "status_code"],
)

HTTP_REQUEST_LATENCY = Histogram(
    name="compass_http_request_latency_seconds",
    documentation="HTTP request latency in seconds per service and endpoint path.",
    labelnames=["service", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

CORRECTION_LOOP_RETRIES = Counter(
    name="compass_correction_loop_retries_total",
    documentation=(
        "Total number of query-reformulation / self-correction retry iterations "
        "triggered by the CorrectionRouterGraph."
    ),
    labelnames=["service", "tenant_id"],
)

PIPELINE_STAGE_DURATION = Histogram(
    name="compass_pipeline_stage_duration_seconds",
    documentation=(
        "Duration in seconds spent inside each named pipeline stage "
        "(e.g. retrieval, contradiction_check, groundedness_check, generation)."
    ),
    labelnames=["service", "stage"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ACTIVE_REQUESTS = Gauge(
    name="compass_active_requests",
    documentation="Number of HTTP requests currently being processed by a service.",
    labelnames=["service"],
)


# ---------------------------------------------------------------------------
# /metrics endpoint wiring
# ---------------------------------------------------------------------------

def setup_metrics(app: Any, service_name: str = "compass-rag") -> None:
    """
    Mount the Prometheus ``/metrics`` endpoint on *app* using
    ``prometheus_fastapi_instrumentator``.

    Call this **after** all routes have been included so the instrumentator
    observes all endpoint paths correctly.

    Args:
        app: FastAPI application instance.
        service_name: Label value injected into automatic HTTP metrics.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            should_instrument_requests_inprogress=True,
            inprogress_name="compass_fastapi_inprogress",
            inprogress_labels=True,
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

        logger.info("Prometheus /metrics endpoint mounted (service=%s).", service_name)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to mount Prometheus /metrics endpoint: %s", exc)
