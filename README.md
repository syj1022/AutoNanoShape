# AutoNanoShape

AutoNanoShape packaged as a Python library with a thin API wrapper around the existing scripts.

## Install

```bash
git clone https://github.com/syj1022/AutoNanoShape.git
cd AutoNanoShape
pip install .
```

Editable (dev) install:

```bash
pip install -e .
```

## Optional Extras

```bash
# dataset creation helpers
pip install ".[data]"

# training helpers
pip install ".[train]"

# prediction helpers
pip install ".[predict]"
```

`predict` installs `torch`, `nff`, and `ase`.

## Quick Start (Python API)

```python
import autonanoshape as ans
```

### Create Dataset

```python
ans.create_dataset(
    input_path="raw_files",
    output_path="dataset_out",
    facets=["MgO"],
)
```

### Train PAINN

```python
ans.train_painn(
    train_path="dataset_out/MgO/train_dataset.pth.tar",
    val_path="dataset_out/MgO/val_dataset.pth.tar",
    test_path="dataset_out/MgO/test_dataset.pth.tar",
    outdir="models/run1",
)
```

### Predict

```python
ans.predict(
    model_path="MLIP/MgO",
    input_path="init.traj",
    output_path="ensemble_results.txt",
)
```

## Build Locally

```bash
python -m pip install --upgrade build
python -m build
```
