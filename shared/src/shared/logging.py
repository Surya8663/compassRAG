import logging
import sys
import time
import uuid
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


def setup_logging(service_name: str, log_level: str = "INFO", json_logs: bool = True) -> None:
    """
    Configure structlog for structured JSON logging across services.
    Every log record will automatically include standard attributes such as timestamp,
    service_name, and any context variables (like request_id).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard logging root
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Inject service_name globally into context vars or processor
    structlog.contextvars.bind_contextvars(service=service_name)

    renderer: Any
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware that extracts or generates an ``X-Request-ID`` header,
    binds it to ``structlog`` context variables (``request_id``), injects the active
    OpenTelemetry ``trace_id`` into log records, and records Prometheus HTTP metrics.
    """

    def __init__(self, app: Any, service_name: str) -> None:
        super().__init__(app)
        self.service_name = service_name
        self.logger = structlog.get_logger(service_name)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Clear any stale context variables from previous requests on the same async worker/thread
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(service=self.service_name)

        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Inject OTel trace_id into structured logs so traces and logs are correlated
        try:
            from shared.telemetry import get_current_trace_id
            structlog.contextvars.bind_contextvars(trace_id=get_current_trace_id())
        except Exception:  # pragma: no cover — graceful degradation
            pass

        start_time = time.perf_counter()
        method = request.method
        url = str(request.url.path)
        client_ip = request.client.host if request.client else "unknown"

        # Increment active requests gauge
        try:
            from shared.metrics import ACTIVE_REQUESTS
            ACTIVE_REQUESTS.labels(service=self.service_name).inc()
        except Exception:  # pragma: no cover
            pass

        self.logger.info(
            "HTTP request started",
            http_method=method,
            http_path=url,
            client_ip=client_ip,
        )

        try:
            response = await call_next(request)
            duration_s = time.perf_counter() - start_time
            duration_ms = round(duration_s * 1000, 2)

            # Ensure X-Request-ID is propagated in the response headers
            response.headers["X-Request-ID"] = request_id

            # Record Prometheus HTTP metrics
            try:
                from shared.metrics import HTTP_REQUEST_LATENCY, HTTP_REQUESTS_TOTAL
                HTTP_REQUESTS_TOTAL.labels(
                    service=self.service_name,
                    method=method,
                    path=url,
                    status_code=str(response.status_code),
                ).inc()
                HTTP_REQUEST_LATENCY.labels(
                    service=self.service_name, path=url
                ).observe(duration_s)
            except Exception:  # pragma: no cover
                pass

            self.logger.info(
                "HTTP request completed",
                http_method=method,
                http_path=url,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self.logger.error(
                "HTTP request failed with unhandled exception",
                http_method=method,
                http_path=url,
                duration_ms=duration_ms,
                exc_info=True,
            )
            raise exc
        finally:
            # Decrement active requests gauge
            try:
                from shared.metrics import ACTIVE_REQUESTS
                ACTIVE_REQUESTS.labels(service=self.service_name).dec()
            except Exception:  # pragma: no cover
                pass
            structlog.contextvars.clear_contextvars()
