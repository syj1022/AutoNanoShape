from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import autonanoshape as ans

ans.predict(
    model_path=str(PROJECT_ROOT / "MLIP" / "forsterite"),
    input_path=str(PROJECT_ROOT / "example" / "init.traj"),
    output_path=str(PROJECT_ROOT / "example" / "ensemble_results.txt"),
    fmax=0.05,
    steps=200,
)
