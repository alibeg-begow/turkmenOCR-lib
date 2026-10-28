# Security policy

## Supported versions

Security fixes are made for the latest released version of turkmenOCR-lib.

## Reporting a vulnerability

Please do not report security problems in public issues. Report them
privately through GitHub instead:
https://github.com/alibeg-begow/turkmenOCR-lib/security/advisories/new

Describe the problem, the steps to reproduce it and the impact you expect.
You will receive a reply within seven days. Once a fix is available, the
problem is disclosed together with the release that fixes it.

## Security notes for users

- turkmenOCR-lib reads weights only in the safetensors format, which cannot
  run code when it is opened.
- The model code that runs with `trust_remote_code=True` is the code bundled
  inside this package, in `turkmenocr/assets/turkmen_ocr/`. Code files placed
  in your weights folder are not executed when the default workspace is used.
- On first use, the tool may install packages with pip. PyTorch comes only
  from https://download.pytorch.org and every other package comes from
  https://pypi.org. Pass `--no-install` to turn this off.
