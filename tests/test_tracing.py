from src.utils.tracing import _normalize_otlp_endpoint


def test_langfuse_otlp_endpoint_normalizes_to_trace_endpoint():
    assert (
        _normalize_otlp_endpoint("https://cloud.langfuse.com/api/public/otel")
        == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    )


def test_langfuse_trace_endpoint_is_left_unchanged():
    assert (
        _normalize_otlp_endpoint("https://cloud.langfuse.com/api/public/otel/v1/traces")
        == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    )
