import importlib.util
import subprocess
import pytest


def test_build_sdist_wheel():
    if importlib.util.find_spec("build.__main__") is None:
        pytest.skip("build module not available in this environment")
    result = subprocess.run(["python", "-m", "build"], check=False)
    assert result.returncode == 0
