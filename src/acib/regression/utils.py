# Copyright © 2026 TU Wien

"""Utilities for regression-stage artifacts."""

from __future__ import annotations

from hashlib import md5
from os.path import relpath
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

_ZENODO_REGRESSION_INPUT_URL = (
    "https://zenodo.org/records/23942533/files/author_regression_input.csv?download=1"
)
_ZENODO_REGRESSION_INPUT_MD5 = "b26890811f6db913e1e6930efc1f1fb3"


def ensure_regression_input_available(
    regression_input_path: Path,
    repo_root: Path,
) -> None:
    """Reuse or download the archived regression input at its canonical path.

    :param regression_input_path: Configured path owned by the regression stage.
    :param repo_root: Repository root used to make status messages relative.
    :raises OSError: If the file cannot be downloaded or written.
    :raises ValueError: If the downloaded file fails its published checksum.
    """
    relative_input_path = relpath(regression_input_path, repo_root)
    if regression_input_path.is_file():
        print(f"Using existing regression input: {relative_input_path}")
        return

    regression_input_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=regression_input_path.parent,
            prefix=f".{regression_input_path.name}.",
            suffix=".download",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            checksum = md5()
            with urlopen(_ZENODO_REGRESSION_INPUT_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
                    checksum.update(chunk)

        if checksum.hexdigest() != _ZENODO_REGRESSION_INPUT_MD5:
            raise ValueError(
                "Downloaded regression input checksum does not match the published "
                "Zenodo file checksum."
            )
        temporary_path.replace(regression_input_path)
        print(f"Downloaded regression input from Zenodo: {relative_input_path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
