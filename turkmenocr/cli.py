"""Command-line entry point.

Standard output carries only the recognised text, so it can be redirected to
a file or piped into another program. Progress notices and diagnostics go to
standard error. Python tracebacks are hidden unless ``--debug`` is given.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List, NoReturn, Optional

from . import __version__
from .diagnostics import EXIT_INTERNAL, EXIT_INTERRUPTED, Reporter, TurkmenOCRError

DESCRIPTION = (
    "Recognise text in an image with the TurkmenOCR model, running locally on "
    "an NVIDIA graphics card with at least 6 GB of video memory."
)

EPILOG = """examples:
  turkmenocr --setup
  turkmenocr -path="/models/turkmen-ocr" -img="page.png"
  turkmenocr --path D:\\models\\turkmen-ocr --img scan.jpg --task markdown > page.md

Run turkmenocr --setup once after installation. It detects the graphics card,
the driver and its CUDA version, and installs the matching GPU build of
PyTorch. The first recognition run does the same if this step is skipped.

The recognised text is written to standard output. Messages about progress
and problems are written to standard error.
"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise TurkmenOCRError(
            "INPUT-000",
            "The command line could not be understood.",
            f"The problem reported was: {message}.",
            "Check the spelling of the options and make sure every path is wrapped "
            'in quotes, for example: turkmenocr -path="C:\\models\\turkmen-ocr" '
            '-img="C:\\scans\\page.png". Run turkmenocr --help to see all options.',
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="turkmenocr",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-path", "--path", dest="path", metavar="DIR",
        help="folder that contains the TurkmenOCR .safetensors weight files",
    )
    parser.add_argument(
        "-img", "--img", dest="img", metavar="FILE",
        help="image to read (PNG, JPEG, BMP, TIFF or WebP)",
    )
    parser.add_argument(
        "--task", choices=["text", "markdown"], default="text",
        help="plain text (default) or a Markdown rendering of the page layout",
    )
    parser.add_argument(
        "--resolution", choices=["tiny", "small", "base", "large", "gundam"], default="gundam",
        help="image detail level; gundam (default) tiles large pages for the best accuracy",
    )
    parser.add_argument("--prompt", help="custom instruction that replaces --task")
    parser.add_argument(
        "--max-new-tokens", type=int, default=4096, metavar="N",
        help="maximum length of the generated text in tokens (default: 4096)",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="detect the graphics card and install the matching GPU build of PyTorch, then exit",
    )
    parser.add_argument("--gpu", type=int, default=0, metavar="INDEX", help="graphics card number to use (default: 0)")
    parser.add_argument(
        "--no-install", action="store_true",
        help="never install or change Python packages; report what is missing instead",
    )
    parser.add_argument(
        "--ignore-vram-check", action="store_true",
        help="skip the 6 GB video memory check (the model may fail to load)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="print only errors and the recognised text")
    parser.add_argument("--debug", action="store_true", help="show full Python tracebacks for errors")
    parser.add_argument("--version", action="version", version=f"turkmenOCR-lib {__version__}")
    return parser


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _auto_install(args: argparse.Namespace) -> bool:
    return not args.no_install and os.environ.get("TURKMENOCR_NO_INSTALL", "") not in ("1", "true", "yes")


def run_setup(args: argparse.Namespace, reporter: Reporter) -> int:
    from .environment import setup_runtime

    summary = setup_runtime(args.gpu, _auto_install(args), reporter)
    reporter.quiet = False
    reporter.info(summary)
    return 0


def run(args: argparse.Namespace, reporter: Reporter) -> int:
    from .engine import TurkmenOCR
    from .validation import normalize_path, validate_image, validate_weights_dir

    weights = validate_weights_dir(normalize_path(args.path))
    image = validate_image(normalize_path(args.img))
    auto_install = _auto_install(args)

    with TurkmenOCR(
        weights,
        gpu=args.gpu,
        auto_install=auto_install,
        ignore_vram_check=args.ignore_vram_check,
        reporter=reporter,
    ) as ocr:
        reporter.info(f"Reading {image.name}.")
        text = ocr.recognize(
            image,
            task=args.task,
            resolution=args.resolution,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _configure_streams()
    debug = "--debug" in (argv if argv is not None else sys.argv[1:])
    reporter = Reporter()
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        reporter.quiet = args.quiet
        if args.setup:
            return run_setup(args, reporter)
        missing = [name for name, value in (("-path/--path", args.path), ("-img/--img", args.img)) if not value]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
        return run(args, reporter)
    except TurkmenOCRError as error:
        if debug:
            traceback.print_exc()
        reporter.error(error)
        return error.exit_code
    except KeyboardInterrupt:
        sys.stderr.write("\nStopped because the user interrupted the program.\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if debug:
            traceback.print_exc()
        reporter.error(
            TurkmenOCRError(
                "INTERNAL-001",
                "An unexpected problem stopped turkmenOCR-lib.",
                f"An error that turkmenOCR-lib does not know how to explain "
                f"occurred: {type(exc).__name__}: {str(exc).strip()[:400]}",
                "Run the same command again with --debug added at the end and "
                "send the complete output to the maintainers at "
                "https://github.com/alibeg-begow/turkmenOCR-lib/issues so the "
                "problem can be fixed.",
            )
        )
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
