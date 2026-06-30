from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def suite_file(tmp_path):
    p = tmp_path / "basic.yaml"
    p.write_text("name: test\ncases: []\n")
    return p


def _patch_pipeline(results=None):
    """Return a context manager that patches run_suite and generate_report."""
    if results is None:
        results = []
    return (
        patch("main.run_suite", return_value=results),
        patch("main.generate_report"),
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_run_requires_suite(runner):
    result = runner.invoke(cli, ["run", "--models", "groq"])
    assert result.exit_code != 0
    assert "suite" in result.output.lower() or "missing" in result.output.lower()


def test_run_requires_models(runner, suite_file):
    result = runner.invoke(cli, ["run", "--suite", str(suite_file)])
    assert result.exit_code != 0
    assert "models" in result.output.lower() or "missing" in result.output.lower()


def test_run_passes_suite_path_to_run_suite(runner, suite_file):
    with patch("main.run_suite", return_value=[]) as mock_run, \
         patch("main.generate_report"):
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq"])
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == suite_file


def test_run_splits_models_on_comma(runner, suite_file):
    with patch("main.run_suite", return_value=[]) as mock_run, \
         patch("main.generate_report"):
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq,gemini,ollama"])
    assert mock_run.call_args[0][1] == ["groq", "gemini", "ollama"]


def test_run_strips_whitespace_from_models(runner, suite_file):
    with patch("main.run_suite", return_value=[]) as mock_run, \
         patch("main.generate_report"):
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq, gemini , ollama"])
    assert mock_run.call_args[0][1] == ["groq", "gemini", "ollama"]


def test_run_single_model(runner, suite_file):
    with patch("main.run_suite", return_value=[]) as mock_run, \
         patch("main.generate_report"):
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq"])
    assert mock_run.call_args[0][1] == ["groq"]


def test_run_default_output_is_reports(runner, suite_file):
    with patch("main.run_suite", return_value=[]), \
         patch("main.generate_report") as mock_report:
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq"])
    assert mock_report.call_args[1]["output_dir"] == "reports"


def test_run_custom_output_dir(runner, suite_file):
    with patch("main.run_suite", return_value=[]), \
         patch("main.generate_report") as mock_report:
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq", "--output", "my-reports"])
    assert mock_report.call_args[1]["output_dir"] == "my-reports"


def test_run_passes_results_to_generate_report(runner, suite_file):
    fake_results = [MagicMock(), MagicMock()]
    with patch("main.run_suite", return_value=fake_results), \
         patch("main.generate_report") as mock_report:
        runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq"])
    assert mock_report.call_args[0][0] is fake_results


# ---------------------------------------------------------------------------
# Output messages
# ---------------------------------------------------------------------------

def test_start_message_contains_suite_name(runner, suite_file):
    with patch("main.run_suite", return_value=[]), patch("main.generate_report"):
        result = runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq"])
    assert suite_file.name in result.output


def test_start_message_contains_all_model_names(runner, suite_file):
    with patch("main.run_suite", return_value=[]), patch("main.generate_report"):
        result = runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq,gemini,ollama"])
    for model in ("groq", "gemini", "ollama"):
        assert model in result.output


def test_completion_message_contains_output_path(runner, suite_file):
    with patch("main.run_suite", return_value=[]), patch("main.generate_report"):
        result = runner.invoke(cli, [
            "run", "--suite", str(suite_file), "--models", "groq", "--output", "out-dir",
        ])
    assert "out-dir" in result.output


def test_successful_run_exits_zero(runner, suite_file):
    with patch("main.run_suite", return_value=[]), patch("main.generate_report"):
        result = runner.invoke(cli, ["run", "--suite", str(suite_file), "--models", "groq"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_suite_file_exits_nonzero(runner, tmp_path):
    missing = str(tmp_path / "nonexistent.yaml")
    result = runner.invoke(cli, ["run", "--suite", missing, "--models", "groq"])
    assert result.exit_code != 0


def test_missing_suite_file_prints_clear_error(runner, tmp_path):
    missing = str(tmp_path / "nonexistent.yaml")
    result = runner.invoke(cli, ["run", "--suite", missing, "--models", "groq"])
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "not found" in combined.lower() or "error" in combined.lower()


def test_missing_suite_file_does_not_call_run_suite(runner, tmp_path):
    missing = str(tmp_path / "ghost.yaml")
    with patch("main.run_suite") as mock_run:
        runner.invoke(cli, ["run", "--suite", missing, "--models", "groq"])
    mock_run.assert_not_called()
