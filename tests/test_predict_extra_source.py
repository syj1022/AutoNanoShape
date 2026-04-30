import tomllib
from pathlib import Path


def test_predict_extra_uses_neuralforcefield_source():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    predict_deps = data["project"]["optional-dependencies"]["predict"]
    assert any(
        dep.startswith("nff @ git+https://github.com/learningmatter-mit/NeuralForceField.git")
        for dep in predict_deps
    )
