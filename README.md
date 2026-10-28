# turkmenOCR-lib

turkmenOCR-lib runs the TurkmenOCR text recognition model on your own
computer. You point it at a folder that contains the model's `.safetensors`
weight files and at an image, and it prints the text found in the image.

Everything else the model needs (configuration, tokenizer and model code) is
bundled with the library. Before loading anything, the tool checks the weight
files, the graphics card and the installed Python packages. It fixes what it
can on its own and explains in plain language how to fix anything it cannot.

## Requirements

Hardware:

- An NVIDIA graphics card with at least 6 GB of video memory, for example a
  GeForce GTX 1060 6 GB, RTX 2060, RTX 3060 or newer. With 8 GB or more, the
  whole model usually fits on the graphics card. With less, part of the model runs on
  the processor automatically, which is slower but gives the same result.
- Cards from the RTX 30 series onward use bfloat16 precision. Older cards
  (GTX 10 and RTX 20 series) use float16 automatically.
- At least 16 GB of system memory.
- About 7 GB of free disk space for the weights, plus about 3 GB for PyTorch.

Software:

- Windows 10 or 11, or Linux.
- The official NVIDIA driver. Running `nvidia-smi` in a terminal must print a
  table that lists your graphics card.
- Python 3.9 to 3.13. Python 3.12 is recommended.

## Installation

```
pip install turkmenOCR-lib
```

We recommend installing into a virtual environment:

```
python -m venv turkmenocr-env
turkmenocr-env\Scripts\activate          (Windows)
source turkmenocr-env/bin/activate       (Linux)
pip install turkmenOCR-lib
```

On first use, the tool installs the GPU build of PyTorch that matches your
driver and graphics card (a download of about 3 GB). It also makes sure
`transformers` 4.46 is installed. This happens once. To install these
packages yourself, see "Installing PyTorch manually" below.

## Preparing the weights

Put the TurkmenOCR weight files in a folder of their own, for example
`D:\models\turkmen-ocr`. The folder needs nothing except the `.safetensors`
file or files. If the weights are split into several parts, such as
`model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`,
every part must be present.

The tool never changes or deletes files in this folder.

## Usage

```
turkmenocr -path="D:\models\turkmen-ocr" -img="C:\scans\page.png"
```

Both single-dash and double-dash option names work, with or without `=`:

```
turkmenocr --path /models/turkmen-ocr --img page.png
```

Only the recognised text goes to standard output, so you can save it to a
file:

```
turkmenocr -path="/models/turkmen-ocr" -img="page.png" > page.txt
```

Progress messages and errors go to standard error, so they appear on screen
but never end up in the saved file.

### Options

- `-path DIR`, `--path DIR`: Folder that contains the `.safetensors` weight files. Required.
- `-img FILE`, `--img FILE`: Image to read: PNG, JPEG, BMP, TIFF or WebP. Required.
- `--task text|markdown`: `text` (the default) returns plain text. `markdown` returns the page as
  Markdown, including headings and tables.
- `--resolution tiny|small|base|large|gundam`: Level of detail. `gundam` (the default) splits large pages into tiles for
  the best accuracy. `base` or `small` use less video memory and run faster.
- `--prompt TEXT`: A custom instruction for the model, replacing `--task`.
- `--max-new-tokens N`: Maximum length of the generated text. The default is 4096.
- `--gpu INDEX`: Graphics card to use when the computer has more than one. The default is 0.
- `--no-install`: Never install or change Python packages. If something is missing, print
  the command that installs it. The environment variable
  `TURKMENOCR_NO_INSTALL=1` does the same.
- `--ignore-vram-check`: Skip the video memory checks. The model may then fail to load.
- `-q`, `--quiet`: Print only errors and the recognised text.
- `--debug`: Show full Python tracebacks, which help when you report a problem.
- `--version`: Print the installed version.

## Using it from Python

```python
from turkmenocr import TurkmenOCR

with TurkmenOCR("/models/turkmen-ocr") as ocr:
    print(ocr.recognize("page1.png"))
    print(ocr.recognize("page2.png", task="markdown"))
```

The model loads once, when `TurkmenOCR` is created. After that, each call to
`recognize` takes only the time needed to read one image. Problems raise
`turkmenocr.TurkmenOCRError`. Its `code`, `summary`, `explanation` and
`resolution` attributes hold the same information the command-line tool
prints.

For a single image, use the shortcut:

```python
from turkmenocr import recognize

text = recognize("/models/turkmen-ocr", "page.png")
```

## What happens when you run it

1. The weights folder and the image are checked. A missing folder, an
   unreadable or damaged image, or a PDF passed by mistake is reported before
   any work starts.
2. The header of each `.safetensors` file is read without loading the
   weights. This finds missing parts, interrupted downloads and damaged
   files. It also detects weights that belong to a different kind of model,
   such as a chat model or an image generator, and names that model.
3. The graphics card is checked with `nvidia-smi`. If needed, PyTorch with
   GPU support and `transformers` 4.46 are installed. The card's total and
   free video memory are compared with what the model needs, and the
   precision is chosen to suit the card.
4. A temporary workspace combines the bundled model files with links to your
   weights. It is created in your user cache folder and removed when the run
   finishes.
5. The model is loaded onto the graphics card, the image is read, and the
   text is printed.

## Exit codes

- 0: success.
- 2: a problem with the command line, the weights folder path or the image.
- 3: a problem with the weight files themselves.
- 4: a problem with the graphics card, the driver or the Python packages.
- 5: the workspace could not be created.
- 6: the model failed while loading or while reading the image.
- 70: an unexpected internal error.
- 130: the program was interrupted with Ctrl+C.

## Troubleshooting

Every error message starts with a code in square brackets, such as
`Error [ENV-003]`, followed by an explanation and the steps that fix it. The
most common cases:

- `ENV-001` (no working NVIDIA graphics card was found): install the latest
  driver from https://www.nvidia.com/Download/index.aspx, restart, and
  confirm that `nvidia-smi` lists your card.
- `ENV-003` (the graphics card has less than 6 GB of video memory):
  TurkmenOCR does not fit on this card. Use a computer with a larger card.
- `ENV-004` (not enough free video memory): close browsers, games and other
  programs that use the graphics card. The `nvidia-smi` command lists the
  programs that are using video memory.
- `RUNTIME-001` (out of memory while reading an image): add
  `--resolution base` or `--resolution small`, or use a smaller image.
- `WEIGHTS-002`, `WEIGHTS-004`, `WEIGHTS-005` (missing or damaged weight
  files): download the file named in the message again.
- `MODEL-001` (the folder holds a different model): point `-path` at the
  folder with the TurkmenOCR weights.
- `WORKSPACE-001` (the tool could not link to the weights, Windows only):
  turn on Developer Mode (Settings, Privacy and security, For developers), or
  move the weights to the same drive as your user folder.

### Installing PyTorch manually

If automatic installation is not possible, for example on a computer without
internet access, install the packages yourself before running the tool. For
most NVIDIA cards with a current driver:

```
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==4.46.0
```

For RTX 50 series cards, use `cu128` instead of `cu126`. With a driver older
than CUDA 12.6, use `cu118`.

### Cache location

The workspace is created under `%LOCALAPPDATA%\turkmenocr` on Windows and
under `~/.cache/turkmenocr` on Linux. Set the `TURKMENOCR_CACHE_DIR`
environment variable to use a different folder.

## Development

```
git clone https://github.com/alibeg-begow/turkmenOCR-lib
cd turkmenOCR-lib
python -m pip install -e ".[dev]"
ruff check .
pytest
```

The tests do not need a graphics card, PyTorch or the model weights. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the library is built and
[docs/RELEASING.md](docs/RELEASING.md) for how releases are published.

## License

turkmenOCR-lib is released under the MIT license. The bundled model support
files in `turkmenocr/assets/turkmen_ocr/` are distributed under the MIT
license included in that folder.
