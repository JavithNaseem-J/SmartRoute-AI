"""
OpenTelemetry tracing — supports generic OTLP backends.

Set OTEL_EXPORTER_OTLP_PROTOCOL to "http/protobuf" or "grpc" (default).
Authentication can be configured natively via OTEL_EXPORTER_OTLP_HEADERS.

Falls back to a no-op when OTEL_EXPORTER_OTLP_ENDPOINT is absent.
"""

import os

from src.utils.logger import logger

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _build_exporter(endpoint: str):
    """Return the right OTEL exporter based on OTEL_EXPORTER_OTLP_PROTOCOL or endpoint URL."""
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "").lower()

    # Parse headers from OTEL_EXPORTER_OTLP_HEADERS
    headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    headers = {}
    if headers_str:
        for item in headers_str.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                headers[k.strip()] = v.strip()

    # Auto-generate Basic Auth for Langfuse if keys are present
    if not headers and ("langfuse" in endpoint.lower() or os.getenv("LANGFUSE_PUBLIC_KEY")):
        import base64

        pub_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sec_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if pub_key and sec_key:
            auth = base64.b64encode(f"{pub_key}:{sec_key}".encode()).decode()
            headers["Authorization"] = f"Basic {auth}"

    kwargs = {"endpoint": endpoint}
    if headers:
        kwargs["headers"] = headers

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCExporter,
        )

        return GRPCExporter(**kwargs)
    else:
        # Default to http/protobuf for HTTP endpoints (e.g., Langfuse / OTLP HTTP)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        return HTTPExporter(**kwargs)


def setup_tracing(app=None) -> None:
    """Configure OTEL tracing and instrument FastAPI.

    Call once after creating the FastAPI app instance.
    No-op if OTEL_EXPORTER_OTLP_ENDPOINT is not set.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("Tracing disabled — set OTEL_EXPORTER_OTLP_ENDPOINT to enable.")
        return

    if not _OTEL_AVAILABLE:
        logger.warning(
            "opentelemetry packages not installed. "
            "Run: pip install opentelemetry-sdk opentelemetry-instrumentation-fastapi "
            "opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-grpc"
        )
        return

    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "smartroute-ai"),
            "service.version": "2.1.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "production"),
        }
    )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(_build_exporter(endpoint)))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
        logger.info(f"Tracing → {endpoint}")
