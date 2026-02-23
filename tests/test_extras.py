import tomllib
from pathlib import Path


def test_extras_defined():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "train" in extras
    assert "predict" in extras
