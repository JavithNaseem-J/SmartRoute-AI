"""
OpenTelemetry tracing setup for SmartRoute-AI.

Instruments FastAPI requests automatically (spans per endpoint).
Exports traces to an OTLP-compatible backend (Jaeger, Tempo, Datadog, etc.)
when OTEL_EXPORTER_OTLP_ENDPOINT is set.

Falls back to a no-op (zero overhead) when the env var is absent or
opentelemetry packages are not installed.

Usage:
    from src.utils.tracing import setup_tracing
    setup_tracing(app)   # call once after creating the FastAPI app
"""
import os
import logging

logger = logging.getLogger("SmartRouteAILogger")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def setup_tracing(app=None, engine=None) -> None:
    """Configure OpenTelemetry tracing.

    Args:
        app: FastAPI application instance (instruments all HTTP requests).
        engine: SQLAlchemy engine (instruments all DB queries).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not endpoint:
        logger.info("Tracing: OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled.")
        return

    if not _OTEL_AVAILABLE:
        logger.warning(
            "Tracing: opentelemetry packages not installed. "
            "Install with: pip install opentelemetry-sdk opentelemetry-instrumentation-fastapi "
            "opentelemetry-instrumentation-sqlalchemy opentelemetry-exporter-otlp-proto-grpc"
        )
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "smartroute-ai"),
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "production"),
    })

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
        logger.info(f"Tracing: FastAPI instrumented → {endpoint}")

    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine)
        logger.info("Tracing: SQLAlchemy instrumented")
