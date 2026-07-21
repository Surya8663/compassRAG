"""
Shared OpenTelemetry telemetry setup for Compass RAG services.

Provides:
- `setup_telemetry(service_name)` — initialise the OTLP-HTTP tracer provider once at startup.
- `get_tracer(name)` — return a named tracer for a module.
- `traced_span(tracer, span_name, **attrs)` — async-safe context manager for explicit child spans.

When `OTEL_ENABLED=false` (or the exporter is unreachable) every call degrades gracefully to
a no-op tracer, so services stay healthy without a running Jaeger instance.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state — initialised once by setup_telemetry()
# ---------------------------------------------------------------------------
_tracer_provider: Any = None


def setup_telemetry(
    service_name: str,
    otlp_endpoint: str = "http://localhost:4318",
    enabled: bool = True,
) -> None:
    """
    Configure the global OpenTelemetry TracerProvider with an OTLP-HTTP exporter.
    Should be called **once** at application startup, before any spans are created.

    Args:
        service_name: Human-readable service identifier (e.g. ``compass-rag-retrieval``).
        otlp_endpoint: Base URL of the OTLP-HTTP collector
            (Jaeger / Grafana Agent / OTEL Collector).
        enabled: When *False* a no-op provider is installed so instrumented code still runs.
    """
    global _tracer_provider  # noqa: PLW0603

    if not enabled:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        _tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        from opentelemetry import trace
        trace.set_tracer_provider(_tracer_provider)
        logger.info("OTel tracing disabled — no-op tracer provider installed for %s", service_name)
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        exporter = OTLPSpanExporter(
            endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces",
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        logger.info(
            "OTel OTLP-HTTP tracer provider initialised for service '%s' -> %s",
            service_name,
            otlp_endpoint,
        )
    except Exception as exc:  # pragma: no cover — only fires if SDK missing
        logger.warning("Failed to initialise OTel tracer provider: %s", exc)


def instrument_fastapi(app: Any) -> None:
    """
    Apply OpenTelemetry auto-instrumentation to a FastAPI app instance.
    Creates HTTP server spans for every incoming request automatically.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.debug("FastAPI OTel instrumentation applied.")
    except Exception as exc:  # pragma: no cover
        logger.warning("FastAPI OTel instrumentation failed: %s", exc)


def get_tracer(name: str) -> Any:
    """
    Return a named :class:`opentelemetry.trace.Tracer` for ``name``.
    Falls back gracefully to the no-op tracer if the SDK is not configured.
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:  # pragma: no cover
        from opentelemetry import trace
        return trace.get_tracer(__name__)


@contextmanager
def traced_span(
    tracer: Any,
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """
    Synchronous context manager that creates an OTel child span.

    Usage::

        with traced_span(tracer, "compass.retrieval.rerank", {"chunk_count": 5}) as span:
            span.set_attribute("result_count", len(results))
            ...
    """
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


@asynccontextmanager
async def async_traced_span(
    tracer: Any,
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """
    Async context manager variant of :func:`traced_span`.

    Usage::

        async with async_traced_span(tracer, "compass.correction.groundedness_check") as span:
            ...
    """
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def get_current_trace_id() -> str:
    """
    Return the hex trace-ID of the currently active span, or ``"0"`` if none.
    Useful for injecting into structured log records.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:  # pragma: no cover
        pass
    return "0"
