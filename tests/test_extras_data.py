import tomllib
from pathlib import Path


def test_data_extra_defined():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "data" in extras
    assert "torch" in extras["data"]
    assert "nff" in extras["data"]
