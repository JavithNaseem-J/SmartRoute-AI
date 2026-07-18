"""
OpenTelemetry tracing — supports both generic OTLP backends and LangFuse.

LangFuse uses HTTP/protobuf (not gRPC), so we auto-detect the endpoint
and pick the right exporter. This means a single env var configures everything:

    OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...

Falls back to a no-op when the env var is absent.
"""

import base64
import os
import logging

logger = logging.getLogger("SmartRouteAILogger")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _build_exporter(endpoint: str):
    """Return the right OTEL exporter based on the endpoint URL.

    LangFuse uses HTTP/protobuf; most self-hosted backends use gRPC.
    We detect LangFuse by checking for 'langfuse' in the URL.
    """
    is_langfuse = "langfuse" in endpoint.lower()

    if is_langfuse:
        # LangFuse requires HTTP Basic Auth encoded as a header.
        # Public key = username, Secret key = password.
        pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sec = os.getenv("LANGFUSE_SECRET_KEY", "")
        token = base64.b64encode(f"{pub}:{sec}".encode()).decode()
        headers = {"Authorization": f"Basic {token}"}

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        return HTTPExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCExporter,
        )

        return GRPCExporter(endpoint=endpoint)


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
