from pathlib import Path

import numpy as np
import pandas as pd

from .config import Folder, SessionConfig
from .registry import DEFAULT_SCORES


def create_demo(root: str | Path) -> SessionConfig:
    root = Path(root).resolve()
    random = np.random.default_rng(42)
    config = SessionConfig(registry=DEFAULT_SCORES.copy(), bootstrap_samples=100)
    experiments = []
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    for target in ["Target-A", "Target-B"]:
        sequences = ["".join(random.choice(alphabet, size=32)) for _ in range(24)]
        strengths = random.uniform(0, 1, len(sequences))
        for model in ["af2", "opendde"]:
            folder = root / target / model
            stats = folder / "traj_stats"
            stats.mkdir(parents=True, exist_ok=True)
            rows2, rows3 = [], []
            for index, (sequence, strength) in enumerate(zip(sequences, strengths)):
                for seed in range(3):
                    name = f"design_{index}_seed_{seed}"
                    noise = random.normal(0, 0.12)
                    rows2.append({"sequence": sequence, "pdbname": name, "iptm": np.clip(0.2 + 0.65 * strength + noise, 0, 1),
                                  "ipsae": np.clip(strength + random.normal(0, 0.15), 0, 1),
                                  "plddt": np.clip(65 + strength * 25 + random.normal(0, 5), 0, 100),
                                  "pae": max(1, 22 - strength * 18 + random.normal(0, 2)),
                                  "novel_score": strength + random.normal(0, 0.2)})
                    rows3.append({"sequence": sequence, "description": f"{name}_0001", "total_score": -100 - 50 * strength + random.normal(0, 20),
                                  "interface_dG": -5 - strength * 35 + random.normal(0, 8),
                                  "interface_sc": np.clip(0.3 + strength * 0.4 + random.normal(0, 0.08), 0, 1)})
            pd.DataFrame(rows2).to_csv(stats / "demo_phase2_stats.csv", index=False)
            pd.DataFrame(rows3).to_csv(stats / "demo_phase3_stats.csv", index=False)
            config.folders.append(Folder(str(folder), target, model))
        for index, (sequence, strength) in enumerate(zip(sequences[:16], strengths[:16])):
            label = "none" if strength < 0.4 else "weak" if strength < 0.6 else "medium" if strength < 0.8 else "strong"
            experiments.append({"target": target, "sequence": sequence, "binding_class": label,
                                "kd_nM": ">10000 nM" if label == "none" else str(round(10 ** (4 - strength * 4), 2)),
                                "fsec_expression": round(random.uniform(0.2, 1), 2)})
    experiment_path = root / "internal.csv"
    pd.DataFrame(experiments).to_csv(experiment_path, index=False)
    config.experiments = [{"path": str(experiment_path), "source": "internal"}]
    (root / "session.yaml").write_text(config.to_yaml())
    return config