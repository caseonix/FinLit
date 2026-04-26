"""Both launch paths build the same app and call serve()."""
import importlib
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from finlit.cli.main import app as cli_app


def test_python_m_main_calls_serve():
    """The __main__.main() function calls serve() with env-derived defaults."""
    # Drop a cached __main__ module so import re-runs (no side effects desired).
    sys.modules.pop("finlit.integrations.mcp.__main__", None)

    with patch("finlit.integrations.mcp.server.serve") as mock_serve:
        # Import is safe because __main__.py only fires serve() under
        # `if __name__ == '__main__'`. main() is the explicit entry point.
        mod = importlib.import_module("finlit.integrations.mcp.__main__")
        mod.main()

    mock_serve.assert_called_once()
    kwargs = mock_serve.call_args.kwargs
    assert "extractor" in kwargs
    assert "pii_mode" in kwargs


def test_finlit_mcp_serve_cli_calls_serve():
    runner = CliRunner()
    with patch("finlit.integrations.mcp.server.serve") as mock_serve:
        result = runner.invoke(cli_app, [
            "mcp", "serve", "--extractor", "ollama", "--pii-mode", "raw",
        ])
    assert result.exit_code == 0, result.output
    mock_serve.assert_called_once()
    kwargs = mock_serve.call_args.kwargs
    assert kwargs["extractor"] == "ollama"
    assert kwargs["pii_mode"] == "raw"


def test_finlit_mcp_serve_rejects_bad_pii_mode():
    runner = CliRunner()
    with patch("finlit.integrations.mcp.server.serve") as mock_serve:
        result = runner.invoke(cli_app, [
            "mcp", "serve", "--pii-mode", "neither",
        ])
    assert result.exit_code != 0
    mock_serve.assert_not_called()
