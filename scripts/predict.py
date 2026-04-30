import sys
import os
import torch
import numpy as np
from pathlib import Path
from nff.io.ase_calcs import NeuralFF, EnsembleNFF
from nff.io.ase import AtomsBatch
from ase.optimize import BFGS
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from ase.io import read
from glob import glob

from autonanoshape.predict_utils import summarize_uncertainty


def predict(
    *,
    model_dir: str,
    input_traj: str,
    output_path: str = "ensemble_results.txt",
    fmax: float = 0.05,
    steps: int = 200,
):
    device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}" if torch.cuda.is_available() else "cpu"

    nnids = [str(i) for i in range(1, 4)]
    model_dirs = [os.path.join(model_dir, str(x), "best_model") for x in nnids]

    models = [NeuralFF.from_file(modeldir, device=device).model for modeldir in model_dirs]

    calc = EnsembleNFF(models, device=device)

    energy_dict = {
        "Mg": -1092.297413,
        "Ca": -1029.314503,
        "O": -460.868328,
        "Si": -168.304750,
        "H": -16.471143,
    }

    atoms = read(input_traj)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    cell = atoms.get_cell()

    bulk = AtomsBatch(
        positions=positions,
        numbers=numbers,
        cell=cell,
        pbc=True,
        cutoff=5.0,
        props={"energy": 0, "energy_grad": []},
        calculator=calc,
        directed=True,
        device=device,
    )
    bulk.update_nbr_list()

    bulk.set_calculator(calc)

    opt = BFGS(bulk)
    opt.run(fmax=fmax, steps=steps)

    energy_std = calc.results.get("energy_std", None)
    forces_std = calc.results.get("forces_std", None)
    uncertainty = summarize_uncertainty(energy_std, forces_std, natoms=len(bulk))

    symbols = bulk.get_chemical_symbols()
    num_O = symbols.count("O")
    num_Mg = symbols.count("Mg")
    num_Ca = symbols.count("Ca")
    num_H = symbols.count("H")
    num_Si = symbols.count("Si")

    corrected_energy = float(
        bulk.get_potential_energy()
        + (
            num_O * energy_dict["O"]
            + num_Mg * energy_dict["Mg"]
            + num_Ca * energy_dict["Ca"]
            + num_H * energy_dict["H"]
            + num_Si * energy_dict["Si"]
        )
    )

    summary = {
        "raw_energy": corrected_energy,
        "energy_std_mev_per_atom": uncertainty["energy_std_mev_per_atom"],
        "mean_force_std": uncertainty["mean_force_std"],
    }

    with open(output_path, "a") as f:
        f.write(f"Raw energy = {corrected_energy:.6f} eV\n")
        f.write(uncertainty["lines"][0] + "\n")
        f.write(uncertainty["lines"][1] + "\n")
        f.write("\n")

    return summary


if __name__ == "__main__":
    predict(model_dir="/path/to/MLIP/MgO", input_traj="init.traj")
