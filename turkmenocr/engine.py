"""Model loading and OCR inference."""

from __future__ import annotations

import contextlib
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Union

from .diagnostics import Reporter, TurkmenOCRError, format_size
from .environment import RuntimeSettings, prepare_environment
from .inspector import WeightsReport, inspect_directory
from .validation import normalize_path, validate_image, validate_weights_dir
from .workspace import prepare_workspace

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

PROMPTS: Dict[str, str] = {
    "text": "<image>\nFree OCR. ",
    "markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
}

MAX_IMAGE_SIDE = 4096


@dataclass(frozen=True)
class Resolution:
    base_size: int
    image_size: int
    crop_mode: bool


RESOLUTIONS: Dict[str, Resolution] = {
    "tiny": Resolution(512, 512, False),
    "small": Resolution(640, 640, False),
    "base": Resolution(1024, 1024, False),
    "large": Resolution(1280, 1280, False),
    "gundam": Resolution(1024, 640, True),
}

_GROUNDING = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>\n?", re.DOTALL)
_SPECIAL = re.compile(r"<｜[^｜]*｜>|<\|(?:ref|/ref|det|/det|grounding)\|>")


def clean_output(text: str) -> str:
    """Remove layout tags and special tokens from generated text."""
    text = _GROUNDING.sub("", text)
    text = _SPECIAL.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_image(source: Union[str, Path, "PILImage"]) -> "PILImage":
    """Open an image as RGB, honour EXIF rotation and cap its size."""
    from PIL import Image, ImageOps

    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        image = background
    else:
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    return image


@contextlib.contextmanager
def _quiet_runtime() -> Iterator[None]:
    """Keep library chatter off stdout, which is reserved for OCR text."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr):
        warnings.simplefilter("ignore")
        yield


class TurkmenOCR:
    """Loads TurkmenOCR once and recognises text in any number of images.

    Example::

        from turkmenocr import TurkmenOCR

        with TurkmenOCR("/models/turkmen-ocr") as ocr:
            print(ocr.recognize("page.png"))
    """

    def __init__(
        self,
        weights_dir: Union[str, Path],
        *,
        gpu: int = 0,
        auto_install: bool = True,
        ignore_vram_check: bool = False,
        reporter: Optional[Reporter] = None,
    ) -> None:
        self.reporter = reporter or Reporter(quiet=True)
        directory = validate_weights_dir(normalize_path(str(weights_dir)))

        self.reporter.info(f"Inspecting model weights in {directory}.")
        self.report: WeightsReport = inspect_directory(directory, self.reporter)
        self.reporter.info(
            f"Found {self.report.variant.display_name} weights: "
            f"{self.report.parameter_count / 1e9:.2f} billion parameters in "
            f"{len(self.report.files)} file(s), {format_size(self.report.total_file_size)}."
        )

        self.settings: RuntimeSettings = prepare_environment(
            required_bytes=self.report.memory_required(bytes_per_parameter=2),
            device_index=gpu,
            auto_install=auto_install,
            ignore_vram_check=ignore_vram_check,
            reporter=self.reporter,
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._load()

    def _load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        dtype = getattr(torch, self.settings.dtype_name)
        self.reporter.info(
            f"Loading the model onto {self.settings.gpu_name} in {self.settings.dtype_name} "
            f"precision ({format_size(self.settings.free_bytes)} of video memory free)."
        )
        with prepare_workspace(self.report, self.reporter) as workspace, _quiet_runtime():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(workspace.path), trust_remote_code=True)
                self._model = AutoModel.from_pretrained(
                    str(workspace.path),
                    trust_remote_code=True,
                    use_safetensors=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map={"": self.settings.device_index},
                    attn_implementation="eager",
                ).eval()
            except torch.cuda.OutOfMemoryError as exc:
                self.close()
                raise self._out_of_memory_error(during="loading the model") from exc
            except Exception as exc:
                self.close()
                raise TurkmenOCRError(
                    "RUNTIME-002",
                    "The model could not be loaded.",
                    f"The weights in {self.report.directory} passed every check, "
                    "but loading them into the model failed with this message: "
                    f"{_first_line(exc)}",
                    "Make sure no other program is modifying the weight files, then "
                    "run the command again. If the problem continues, download the "
                    "weights again, because one of the files may have been altered "
                    "after it was saved.",
                ) from exc

    def recognize(
        self,
        image: Union[str, Path, "PILImage"],
        *,
        task: str = "text",
        resolution: str = "gundam",
        prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
    ) -> str:
        """Return the text found in an image.

        Args:
            image: Path to an image file, or an already opened PIL image.
            task: ``"text"`` for plain text or ``"markdown"`` for a Markdown
                rendering of the page layout.
            resolution: One of ``tiny``, ``small``, ``base``, ``large`` or
                ``gundam`` (tiled high resolution, the default).
            prompt: Custom instruction overriding ``task``.
            max_new_tokens: Upper bound on the length of the generated text.
        """
        import torch

        if self._model is None:
            raise RuntimeError("This TurkmenOCR instance has been closed.")
        if task not in PROMPTS:
            raise ValueError(f"task must be one of {sorted(PROMPTS)}")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
        if isinstance(image, (str, Path)):
            image = validate_image(normalize_path(str(image)))
        text_prompt = prompt if prompt is not None else PROMPTS[task]
        if "<image>" not in text_prompt:
            text_prompt = "<image>\n" + text_prompt
        mode = RESOLUTIONS[resolution]

        picture = prepare_image(image)
        try:
            with _quiet_runtime(), torch.inference_mode():
                raw = self._model.infer(
                    self._tokenizer,
                    prompt=text_prompt,
                    pil_image=picture,
                    base_size=mode.base_size,
                    image_size=mode.image_size,
                    crop_mode=mode.crop_mode,
                    max_new_tokens=max_new_tokens,
                    eval_mode=True,
                )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise self._out_of_memory_error(during="reading the image") from exc
        except Exception as exc:
            raise TurkmenOCRError(
                "RUNTIME-003",
                "Text recognition failed.",
                f"The model was loaded successfully, but processing the image "
                f"failed with this message: {_first_line(exc)}",
                "Try the command again with a different image, or save this image "
                "again as PNG and retry. If it keeps failing, run the command with "
                "--debug and report the full output to the maintainers.",
            ) from exc
        return clean_output(raw or "")

    def _out_of_memory_error(self, during: str) -> TurkmenOCRError:
        free, total = _memory_figures(self.settings.device_index)
        return TurkmenOCRError(
            "RUNTIME-001",
            f"The graphics card ran out of memory while {during}.",
            f"The graphics card {self.settings.gpu_name} has {format_size(total)} "
            f"of video memory, of which {format_size(free)} is free now. While "
            f"{during}, TurkmenOCR needed more memory than was available, so the "
            "work was stopped. Very large or very detailed images and other "
            "programs using the graphics card both increase memory use.",
            "Close other programs that use the graphics card, such as web "
            "browsers, games and other AI tools. If the problem continues, use a "
            "smaller image or add --resolution base (or small) to the command so "
            "that the image is processed in fewer pieces.",
        )

    def close(self) -> None:
        """Release the model and free its video memory."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> "TurkmenOCR":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def recognize(weights_dir: Union[str, Path], image: Union[str, Path, "PILImage"], **options: Any) -> str:
    """Load the model, read one image and release the model again."""
    engine_keys = ("gpu", "auto_install", "ignore_vram_check", "reporter")
    engine_options = {key: options.pop(key) for key in engine_keys if key in options}
    with TurkmenOCR(weights_dir, **engine_options) as ocr:
        return ocr.recognize(image, **options)


def _memory_figures(device_index: int) -> "tuple[int, int]":
    try:
        import torch

        free, total = torch.cuda.mem_get_info(device_index)
        return int(free), int(total)
    except Exception:
        return 0, 0


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text.splitlines()[0][:400]
