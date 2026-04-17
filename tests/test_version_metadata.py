import json
from pathlib import Path

import tomllib

from hermes_shim_http import __version__


def test_release_metadata_versions_are_aligned():
    repo_root = Path(__file__).resolve().parents[1]
    package_version = json.loads((repo_root / "package.json").read_text())["version"]
    pyproject_version = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"]["version"]

    assert package_version == pyproject_version == __version__
