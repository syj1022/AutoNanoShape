from typing import Any, Dict, List, Optional

import numpy as np


def summarize_uncertainty(
    energy_std: Optional[np.ndarray],
    forces_std: Optional[np.ndarray],
    *,
    natoms: int,
) -> Dict[str, Any]:
    energy_std_mev_per_atom = None
    mean_force_std = None

    if energy_std is not None:
        energy_std_mev_per_atom = float(np.mean(energy_std / natoms * 1000))

    if forces_std is not None:
        mean_force_std = float(np.mean(np.linalg.norm(forces_std, axis=1)))

    lines: List[str] = []
    if energy_std_mev_per_atom is None:
        lines.append("Energy s.t.d. = N/A")
    else:
        lines.append(f"Energy s.t.d. = {energy_std_mev_per_atom:.6f} meV/atom")

    if mean_force_std is None:
        lines.append("Mean force s.t.d. = N/A")
    else:
        lines.append(f"Mean force s.t.d. = {mean_force_std:.6f} eV/Å")

    return {
        "energy_std_mev_per_atom": energy_std_mev_per_atom,
        "mean_force_std": mean_force_std,
        "lines": lines,
    }
