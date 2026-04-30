from pathlib import Path

from autonanoshape import api


def test_nvidia_runtime_library_dirs_prioritize_env_site_packages(tmp_path, monkeypatch):
    env_prefix = tmp_path / "env"
    py_tag = f"python{api.sys.version_info.major}.{api.sys.version_info.minor}"
    nvidia_root = env_prefix / "lib" / py_tag / "site-packages" / "nvidia"
    for relative in (
        "nvjitlink/lib",
        "cusparse/lib",
        "cublas/lib",
        "cuda_runtime/lib",
        "cuda_nvrtc/lib",
    ):
        (nvidia_root / relative).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(api.sys, "prefix", str(env_prefix))
    monkeypatch.setattr(api.sys, "base_prefix", str(tmp_path / "base"))

    paths = api._nvidia_runtime_library_dirs()

    assert len(paths) >= 5
    assert all(str(p).startswith(str(nvidia_root)) for p in paths[:5])
