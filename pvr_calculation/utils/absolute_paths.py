"""Paths inside the importable ``pvr_calculation`` package directory."""

from __future__ import annotations

import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PACKAGE_ROOT / "data"
BFI_PATH = PACKAGE_ROOT / "data" / "time_series_data"
RESULTS_PATH = PACKAGE_ROOT / "data" / "segmentation_and_pvr_results"
FIGURE_PATH = PACKAGE_ROOT / "data" / "figures"

REPO_PATH = PACKAGE_ROOT


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    """Create a directory when you are about to write (avoid side effects at import)."""
    path.mkdir(parents=True, exist_ok=True)
    return path
