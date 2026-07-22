"""OpenTelemetry instrumentation setup for academic paper system."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


def setup_telemetry(app, endpoint: str = "") -> None:
    """OTelトレースを設定する.

    Args:
        app: FastAPI application instance
        endpoint: OpenTelemetry endpoint (e.g., "http://localhost:4317")
                 If empty, tracing is disabled (NoopTracerProvider)
    """
    if not endpoint:
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str = "academic_paper"):
    """Get a tracer instance.

    Args:
        name: Name of the tracer (default: "academic_paper")

    Returns:
        A tracer instance from the global tracer provider
    """
    return trace.get_tracer(name)
