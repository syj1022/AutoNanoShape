import ctypes
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any, Callable, Dict


def _is_missing_module(exc: ModuleNotFoundError, names: set[str]) -> bool:
    return exc.name in names or any(exc.name.startswith(f"{name}.") for name in names)


def _raise_missing_extra(module_name: str, extra: str, command: str) -> None:
    raise ModuleNotFoundError(
        f"Missing optional dependency '{module_name}'. "
        f"Install {extra} requirements with: {command}"
    ) from None


def _nvidia_runtime_library_dirs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    site_roots = {
        Path(sys.prefix) / "lib",
        Path(sys.prefix) / "Lib",
    }
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        site_roots.update({Path(sys.base_prefix) / "lib", Path(sys.base_prefix) / "Lib"})

    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    relative_nvidia_libs = (
        f"{py_tag}/site-packages/nvidia/nvjitlink/lib",
        f"{py_tag}/site-packages/nvidia/cusparse/lib",
        f"{py_tag}/site-packages/nvidia/cublas/lib",
        f"{py_tag}/site-packages/nvidia/cuda_runtime/lib",
        f"{py_tag}/site-packages/nvidia/cuda_nvrtc/lib",
    )

    for root in site_roots:
        for rel in relative_nvidia_libs:
            path = root / rel
            if path.exists() and path not in seen:
                seen.add(path)
                candidates.append(path)
    return candidates


def _preload_torch_cuda_runtime() -> None:
    for lib_dir in _nvidia_runtime_library_dirs():
        for lib_name in ("libnvJitLink.so.12", "libcusparse.so.12"):
            lib_path = lib_dir / lib_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def _load_script_function(module_file: str, function_name: str) -> Callable[..., Dict[str, Any]]:
    project_root = Path(__file__).resolve().parent.parent
    module_path = project_root / "scripts" / module_file
    if not module_path.exists():
        raise ModuleNotFoundError(
            f"Unable to locate script module at '{module_path}'. "
            f"Run commands from the project checkout root."
        ) from None

    module_key = f"_autonanoshape_scripts_{module_path.stem}"
    spec = spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for '{module_path}'.") from None

    _preload_torch_cuda_runtime()
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def create_dataset(*, input_path: str = "raw_files", output_path: str = ".", **kwargs) -> Dict[str, Any]:
    try:
        _create_dataset = _load_script_function("create_dataset.py", "create_dataset")
    except ModuleNotFoundError as exc:
        if _is_missing_module(exc, {"torch", "nff"}):
            _raise_missing_extra(exc.name, "data", "pip install --force-reinstall '.[data]'")
        raise

    return _create_dataset(raw_dir=input_path, output_dir=output_path, **kwargs)


def train_painn(
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    outdir: str = ".",
    **kwargs,
) -> Dict[str, Any]:
    try:
        _train_painn = _load_script_function("training_painn.py", "train_painn")
    except ModuleNotFoundError as exc:
        if _is_missing_module(exc, {"torch", "nff", "matplotlib"}):
            _raise_missing_extra(exc.name, "train", "pip install --force-reinstall '.[train]'")
        raise

    return _train_painn(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        outdir=outdir,
        **kwargs,
    )


def predict(*, model_path: str, input_path: str, output_path: str = "ensemble_results.txt", **kwargs) -> Dict[str, Any]:
    try:
        _predict = _load_script_function("predict.py", "predict")
    except ModuleNotFoundError as exc:
        if _is_missing_module(exc, {"torch", "nff", "ase"}):
            _raise_missing_extra(exc.name, "predict", "pip install --force-reinstall '.[predict]'")
        raise

    return _predict(model_dir=model_path, input_traj=input_path, output_path=output_path, **kwargs)
