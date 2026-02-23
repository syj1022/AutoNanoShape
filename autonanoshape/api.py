from typing import Any, Dict


def create_dataset(*, input_path: str = "raw_files", output_path: str = ".", **kwargs) -> Dict[str, Any]:
    from scripts.create_dataset import create_dataset as _create_dataset

    return _create_dataset(raw_dir=input_path, output_dir=output_path, **kwargs)


def train_painn(
    *,
    train_path: str,
    val_path: str,
    test_path: str,
    outdir: str = ".",
    **kwargs,
) -> Dict[str, Any]:
    from scripts.training_painn import train_painn as _train_painn

    return _train_painn(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        outdir=outdir,
        **kwargs,
    )


def predict(*, model_path: str, input_path: str, output_path: str = "ensemble_results.txt", **kwargs) -> Dict[str, Any]:
    from scripts.predict import predict as _predict

    return _predict(model_dir=model_path, input_traj=input_path, output_path=output_path, **kwargs)
