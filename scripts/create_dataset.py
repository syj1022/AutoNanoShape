import os
import re
import json
import sys
import torch
import pickle
import numpy as np
from importlib import reload
import networkx as nx
from ast import literal_eval
from torch.utils.data import DataLoader
import nff.data as d
import pandas as pd
from collections import defaultdict
from nff.data.dataset import Dataset, concatenate_dict, split_train_validation_test, stratified_split
from nff.data.stats import get_stoich_dict, perform_energy_offset, remove_dataset_outliers, center_dataset


def data_to_tensor(data):
    converted_matrices = []
    for data_string in data:
        cleaned_data_string = re.sub(r"(?<![\[\],])\s+", ", ", data_string)
        matrix = literal_eval(cleaned_data_string)
        converted_matrices.append(matrix)
    tensor_list = [torch.tensor(matrix) for matrix in converted_matrices]
    return tensor_list


def calculate_energy_correction(formula):
    pattern = r"([A-Z][a-z]*)(\d*)"
    counts = defaultdict(int)
    for element, count in re.findall(pattern, formula):
        counts[element] += int(count) if count else 1
    energy_dict = {
        "Mg": -1092.297413,
        "Ca": -1029.314503,
        "O": -460.868328,
        "Si": -168.304750,
        "H": -16.471143,
    }
    total_energy_correction = sum(count * energy_dict[element] * 23.060542 for element, count in counts.items())
    return total_energy_correction


def create_dataset(
    *,
    facets=None,
    raw_dir="raw_files",
    output_dir=".",
    cutoff=5,
    val_size=0.2,
    test_size=0.1,
    seed=11,
):
    if facets is None:
        facets = ["MgO"]

    results = {}
    for facet in facets:
        file_base = os.path.join(raw_dir, facet)
        data = pd.read_csv(file_base + ".csv")

        formula = data["formula"]
        lattice = data["lattice"].values
        nxyz = data["nxyz"].values
        force = data["force"].values
        energy = data["energy"].values

        np.savez(file_base + ".npz", formula=formula, lattice=lattice, nxyz=nxyz, force=force, energy=energy)

        loaded_data = np.load(file_base + ".npz", allow_pickle=True)

        formula_data = loaded_data.f.formula
        lattice_data = data_to_tensor(loaded_data.f.lattice)
        nxyz_data = data_to_tensor(loaded_data.f.nxyz)
        force_data = data_to_tensor(loaded_data.f.force)
        energy_data = loaded_data.f.energy.squeeze() * 23.060542

        for i in range(len(energy_data)):
            energy_data[i] -= calculate_energy_correction(formula_data[i])

        props = {
            "formula": formula_data.tolist(),
            "lattice": lattice_data,
            "nxyz": nxyz_data,
            "energy": energy_data.tolist(),
            "energy_grad": [(-x) for x in force_data],
        }

        dataset = d.Dataset(props.copy(), units="kcal/mol")
        dataset.generate_neighbor_list(cutoff=cutoff)

        out_facet_dir = os.path.join(output_dir, facet)
        os.makedirs(out_facet_dir, exist_ok=True)
        dataset.save(os.path.join(out_facet_dir, "full_dataset.pth.tar"))

        train_dset, val_dset, test_dset = split_train_validation_test(
            dataset,
            stratified=True,
            targ_name="formula",
            val_size=val_size,
            test_size=test_size,
            seed=seed,
        )

        final_full_dset, final_train_dset, final_val_dset, final_test_dset = (
            dataset.copy(),
            train_dset.copy(),
            val_dset.copy(),
            test_dset.copy(),
        )
        final_dsets = [final_full_dset, final_train_dset, final_val_dset, final_test_dset]

        dset_labels = ["full", "train", "val", "test"]
        for dset, dset_label in zip(final_dsets, dset_labels):
            dset.save(os.path.join(out_facet_dir, f"{dset_label}_dataset.pth.tar"))

        results[facet] = {
            "output_dir": out_facet_dir,
            "paths": {
                "full": os.path.join(out_facet_dir, "full_dataset.pth.tar"),
                "train": os.path.join(out_facet_dir, "train_dataset.pth.tar"),
                "val": os.path.join(out_facet_dir, "val_dataset.pth.tar"),
                "test": os.path.join(out_facet_dir, "test_dataset.pth.tar"),
            },
        }

    return results


if __name__ == "__main__":
    create_dataset()
