import pytest
from PIL import Image

from turkmenocr import __version__
from turkmenocr.cli import build_parser, main
from turkmenocr.diagnostics import EXIT_INPUT, EXIT_WEIGHTS, TurkmenOCRError
from turkmenocr.validation import normalize_path


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (32, 16), "white").save(path)
    return path


def test_single_dash_options_with_equals():
    args = build_parser().parse_args(['-path=/models/ocr', '-img=page.png'])
    assert args.path == "/models/ocr"
    assert args.img == "page.png"


def test_double_dash_options():
    args = build_parser().parse_args(["--path", "/models/ocr", "--img", "page.png", "--task", "markdown"])
    assert args.task == "markdown"
    assert args.resolution == "gundam"


def test_normalize_path_strips_stray_quotes(tmp_path):
    assert normalize_path(f'"{tmp_path}"') == tmp_path.resolve()
    assert normalize_path(f'{tmp_path}"') == tmp_path.resolve()


def test_missing_arguments_are_explained_without_traceback(capsys):
    assert main([]) == EXIT_INPUT
    captured = capsys.readouterr()
    assert captured.err.startswith("Error [INPUT-000]")
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_missing_weights_folder(tmp_path, image, capsys):
    code = main([f"-path={tmp_path / 'nope'}", f"-img={image}"])
    assert code == EXIT_INPUT
    assert "Error [INPUT-001]" in capsys.readouterr().err


def test_weights_path_is_a_file(tmp_path, image, capsys):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"x")
    assert main(["-path", str(weights), "-img", str(image)]) == EXIT_INPUT
    err = capsys.readouterr().err
    assert "Error [INPUT-002]" in err
    assert str(tmp_path) in err


def test_corrupt_image(tmp_path, capsys):
    bad = tmp_path / "page.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nbroken")
    assert main(["-path", str(tmp_path), "-img", str(bad)]) == EXIT_INPUT
    assert "Error [INPUT-006]" in capsys.readouterr().err


def test_pdf_is_explained(tmp_path, capsys):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    assert main(["-path", str(tmp_path), "-img", str(pdf)]) == EXIT_INPUT
    assert "PDF document" in capsys.readouterr().err


def test_wrong_weight_format_reaches_inspector(tmp_path, image, capsys):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.gguf").write_bytes(b"GGUF")
    assert main(["-path", str(weights), "-img", str(image), "-q"]) == EXIT_WEIGHTS
    captured = capsys.readouterr()
    assert "Error [WEIGHTS-001]" in captured.err
    assert "GGUF" in captured.err
    assert captured.out == ""


def test_unexpected_errors_are_wrapped(monkeypatch, tmp_path, image, capsys):
    def boom(args, reporter):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("turkmenocr.cli.run", boom)
    assert main(["-path", str(tmp_path), "-img", str(image)]) == 70
    err = capsys.readouterr().err
    assert "Error [INTERNAL-001]" in err
    assert "kaboom" in err
    assert "Traceback" not in err


def test_debug_shows_traceback(monkeypatch, tmp_path, image, capsys):
    def fail(args, reporter):
        raise TurkmenOCRError("ENV-001", "s", "e", "r")

    monkeypatch.setattr("turkmenocr.cli.run", fail)
    main(["-path", str(tmp_path), "-img", str(image), "--debug"])
    assert "Traceback" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
