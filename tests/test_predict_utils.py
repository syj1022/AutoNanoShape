import numpy as np

from autonanoshape.predict_utils import summarize_uncertainty


def test_summarize_uncertainty_handles_none():
    result = summarize_uncertainty(None, None, natoms=10)
    assert result["energy_std_mev_per_atom"] is None
    assert result["mean_force_std"] is None
    assert result["lines"][0].endswith("N/A")
    assert result["lines"][1].endswith("N/A")


def test_summarize_uncertainty_computes_values():
    energy_std = np.array([1.0, 3.0])
    forces_std = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    result = summarize_uncertainty(energy_std, forces_std, natoms=2)
    assert result["energy_std_mev_per_atom"] == 1000.0
    assert result["mean_force_std"] == 5.0
