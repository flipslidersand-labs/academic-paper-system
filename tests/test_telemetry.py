"""Tests for telemetry module."""

from unittest.mock import MagicMock, patch

from academic_paper.telemetry import get_tracer, setup_telemetry


def test_setup_telemetry_empty_endpoint_is_noop():
    """Test setup_telemetry returns None immediately when endpoint is empty."""
    app = MagicMock()
    result = setup_telemetry(app, "")
    assert result is None
    app.assert_not_called()


def test_setup_telemetry_default_endpoint_is_noop():
    """Test setup_telemetry with default (empty) endpoint is a noop."""
    app = MagicMock()
    result = setup_telemetry(app)
    assert result is None


def test_setup_telemetry_with_endpoint_configures_provider():
    """Test setup_telemetry wires OTel provider when endpoint is given."""
    app = MagicMock()
    mock_otlp_module = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": mock_otlp_module,
        },
    ), patch("academic_paper.telemetry.TracerProvider") as mock_provider_cls, \
       patch("academic_paper.telemetry.BatchSpanProcessor"), \
       patch("academic_paper.telemetry.trace") as mock_trace, \
       patch("academic_paper.telemetry.FastAPIInstrumentor") as mock_fapi, \
       patch("academic_paper.telemetry.HTTPXClientInstrumentor") as mock_httpx_inst:

        mock_provider = MagicMock()
        mock_provider_cls.return_value = mock_provider

        setup_telemetry(app, "http://localhost:4317")

        mock_provider_cls.assert_called_once()
        mock_provider.add_span_processor.assert_called_once()
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)
        mock_fapi.instrument_app.assert_called_once_with(app)
        mock_httpx_inst.return_value.instrument.assert_called_once()


def test_get_tracer_default_name():
    """Test get_tracer returns a non-None tracer with default name."""
    tracer = get_tracer()
    assert tracer is not None


def test_get_tracer_custom_name():
    """Test get_tracer returns a non-None tracer with custom name."""
    tracer = get_tracer("my_custom_service")
    assert tracer is not None
