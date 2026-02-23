# AutoNanoShape

AutoNanoShape packaged as a Python library with a thin API wrapper around the existing scripts.

## Install (GitHub Packages)

```bash
pip install --extra-index-url https://<TOKEN>@pypi.pkg.github.com/<OWNER> autonanoshape
```

Replace:
- `<OWNER>` with your GitHub org/user name
- `<TOKEN>` with a GitHub token that has `read:packages`

## Optional Extras

```bash
# dataset creation helpers
pip install --extra-index-url https://<TOKEN>@pypi.pkg.github.com/<OWNER> "autonanoshape[data]"

# training helpers
pip install --extra-index-url https://<TOKEN>@pypi.pkg.github.com/<OWNER> "autonanoshape[train]"

# prediction helpers
pip install --extra-index-url https://<TOKEN>@pypi.pkg.github.com/<OWNER> "autonanoshape[predict]"
```

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

## Publish (GitHub Packages)

This repo includes `.github/workflows/publish.yml` which publishes on tag pushes.

```bash
git tag v0.1.0
git push --tags
```

## Build Locally

```bash
python -m pip install --upgrade build
python -m build
```
