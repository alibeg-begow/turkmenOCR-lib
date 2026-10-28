from turkmenocr.diagnostics import (
    EXIT_ENVIRONMENT,
    EXIT_INPUT,
    EXIT_INTERNAL,
    EXIT_WEIGHTS,
    Reporter,
    TurkmenOCRError,
    format_error,
    format_size,
)


def make_error(**overrides):
    fields = dict(
        code="ENV-003",
        summary="The graphics card does not have enough memory.",
        explanation="The card has 6.0 GB of video memory in total.",
        resolution="Use a card with 8 GB or more.",
    )
    fields.update(overrides)
    return TurkmenOCRError(**fields)


def test_three_paragraph_layout():
    text = format_error(make_error())
    paragraphs = text.strip().split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[0] == "Error [ENV-003]: The graphics card does not have enough memory."
    assert paragraphs[1].startswith("The card has 6.0 GB")
    assert paragraphs[2].startswith("How to resolve this issue: Use a card")


def test_steps_are_numbered_inside_the_resolution_paragraph():
    text = format_error(make_error(steps=["First step.", "Second step."]))
    paragraphs = text.strip().split("\n\n")
    assert len(paragraphs) == 3
    assert "\n1. First step.\n2. Second step." in paragraphs[2]


def test_output_is_plain_ascii_without_decoration():
    text = format_error(make_error(explanation="x " * 200))
    assert text.isascii()
    assert "\x1b[" not in text
    assert "===" not in text and "---" not in text
    assert all(len(line) <= 79 for line in text.splitlines())


def test_exit_codes_follow_category():
    assert make_error(code="INPUT-001").exit_code == EXIT_INPUT
    assert make_error(code="WEIGHTS-002").exit_code == EXIT_WEIGHTS
    assert make_error(code="MODEL-001").exit_code == EXIT_WEIGHTS
    assert make_error(code="ENV-001").exit_code == EXIT_ENVIRONMENT
    assert make_error(code="SOMETHING-1").exit_code == EXIT_INTERNAL


def test_format_size():
    assert format_size(512) == "512 bytes"
    assert format_size(6 * 1024 ** 3) == "6.0 GB"
    assert format_size(12884901888) == "12.0 GB"


def test_quiet_reporter_still_reports_errors(capsys):
    reporter = Reporter(quiet=True)
    reporter.info("hidden")
    reporter.error(make_error())
    captured = capsys.readouterr()
    assert "hidden" not in captured.err
    assert "Error [ENV-003]" in captured.err
    assert captured.out == ""
