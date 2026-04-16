# mc_analysis.is developed using the version philosophy described here: https://semver.org/
import importlib.metadata
from pathlib import Path

import toml

_PROJECT_PATH = Path(__file__).parents[1]

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    # Try fetching it from pyproject.toml
    try:
        tpath = _PROJECT_PATH / "pyproject.toml"
        __version__ = toml.load(tpath)["tool"]["poetry"]["version"]
    except Exception:
        __version__ = "unknown"

__commit__ = open(_PROJECT_PATH / "commit.txt").read().strip()
