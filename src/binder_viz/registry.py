from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

from .paths import input_file, output_file


METRIC_GROUPS = {
    "refold": {"iptm": 1, "ipsae": 1},
    "electrostatics": {"apbs_elec_complementarity": 1, "apbs_dG_elec": -1,
                       "prodigy_dG_bind": -1, "prodigy_log10_kd": -1},
    "rosetta": {"interface_dG": -1, "interface_sc": 1, "interface_packstat": 1,
                "dG_cross/dSASAx100": -1},
}
GROUP_WEIGHTS = {"refold": 0.50, "electrostatics": 0.30, "rosetta": 0.20}
SHARED_ELEC_COLS = ["apbs_dG_elec", "apbs_dG_coulomb", "apbs_dG_desolv", "apbs_elec_complementarity",
                    "prodigy_dG_bind", "prodigy_kd_nM", "prodigy_log10_kd"]


DEFAULT_SCORES = [
    {"name": name, "stage": stage, "direction": direction, "group": group}
    for stage, group, direction, names in [
        ("phase2", "confidence", "higher_better", ["iptm", "ipsae", "ipsae_binder", "plddt", "ptm"]),
        ("phase2", "confidence", "lower_better", ["pae", "PAE_interaction", "pae_interaction"]),
        ("phase3", "interface", "lower_better", ["dG_separated", "interface_dG", "dG_cross/dSASAx100"]),
        ("phase3", "energy", "lower_better", ["total_score", "score", "binder_score"]),
        ("phase3", "interface", "higher_better", ["SASA", "dSASA_int", "interface_sc", "interface_packstat", "shape_complementarity"]),
    ] for name in names
]
DEFAULT_SCORES += [
    {"name": name, "stage": stage, "direction": "higher_better" if sign == 1 else "lower_better", "group": group}
    for group, metrics in METRIC_GROUPS.items() for name, sign in metrics.items()
    for stage in (["phase2", "phase3"] if group == "electrostatics" else ["phase2"] if group == "refold" else ["phase3"])
    if not any(row["name"] == name and row["stage"] == stage for row in DEFAULT_SCORES)
]
DEFAULT_SCORES += [
    {"name": name, "stage": "derived", "direction": "higher_better", "group": "composite"}
    for name in ["composite", "score_refold", "score_electrostatics", "score_rosetta"]
]
DIRECTIONS = {"higher_better", "lower_better", "unknown"}
DERIVED_SCORES = {"composite", "score_refold", "score_electrostatics", "score_rosetta"}
DERIVED_SCORES.update(f"pct_{name}" for metrics in METRIC_GROUPS.values() for name in metrics)
CONSENSUS_MODES = ["mean_rank", "mean_zscore", "min", "passing"]


def builtin_composites(models: list[str], directions: dict[str, str]) -> list[dict]:
    """Composite and consensus recipes offered for every loaded model, skipped when their ingredients are missing."""
    specs = []
    for model in models:
        confidence = [column for column in [f"iptm__{model}", f"ipsae__{model}"] if column in directions]
        if len(confidence) == 2:
            specs.append({"column": f"composite_confidence__{model}", "kind": "composite",
                          "weights": dict.fromkeys(confidence, 1.0), "normalization": "rank"})
        interface = [f"iptm__{model}", f"ros_interface_dG__{model}"]
        if all(column in directions for column in interface):
            specs.append({"column": f"composite_interface__{model}", "kind": "composite",
                          "weights": dict.fromkeys(interface, 1.0), "normalization": "rank"})
    iptms = [f"iptm__{model}" for model in models if f"iptm__{model}" in directions]
    if len(iptms) > 1:
        specs += [{"column": f"consensus_{mode}__all", "kind": "consensus", "columns": iptms,
                   "mode": mode, "minimum_models": len(iptms)} for mode in CONSENSUS_MODES]
    return specs


def load_registry(path: str | Path) -> list[dict]:
    path = input_file(path, (".csv", ".yaml", ".yml"))
    if path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).replace({np.nan: None}).to_dict("records")
    else:
        data = yaml.safe_load(path.read_text()) or {}
        rows = data.get("scores", []) if isinstance(data, dict) else data
    for row in rows:
        if not {"name", "stage", "direction"}.issubset(row):
            raise ValueError("Registry rows require name, stage and direction.")
        if row["direction"] not in DIRECTIONS or row["stage"] not in {"phase2", "phase3", "derived"}:
            raise ValueError(f"Invalid registry entry: {row}")
    return rows


def save_registry(rows: list[dict], path: str | Path):
    path = output_file(path, "score_registry.yaml")
    if path.suffix.lower() == ".csv":
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.write_text(yaml.safe_dump({"scores": rows}, sort_keys=False))


def infer_direction(values: pd.Series, labels: pd.Series) -> str:
    data = pd.DataFrame({"score": pd.to_numeric(values, errors="coerce"),
                         "label": pd.to_numeric(labels, errors="coerce")}).dropna()
    if len(data) < 3 or data.score.nunique() < 2 or data.label.nunique() < 2:
        return "unknown"
    effect = spearmanr(data.score, data.label).statistic
    if not np.isfinite(effect) or effect == 0:
        return "unknown"
    return "higher_better" if effect > 0 else "lower_better"


def score_catalog(table: pd.DataFrame, registry: list[dict] | None = None,
                  experiments: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    registry = DEFAULT_SCORES if registry is None else registry
    definitions = {(row["stage"], row["name"]): row for row in registry if not row.get("column")}
    specific = {row["column"]: row for row in registry if row.get("column")}
    candidates = [column for column in table if "__" in column and not column.startswith(("n_rep_", "files_", "rank__"))
                  and pd.api.types.is_numeric_dtype(table[column])]
    evidence = None
    if experiments is not None and not experiments.empty and "binding_ord" in experiments:
        labels = experiments.groupby("sequence_id")["binding_ord"].agg(lambda values: values.dropna().iloc[0] if values.dropna().nunique() == 1 else np.nan)
        evidence = table["sequence_id"].map(labels)
    rows = []
    found = set()
    for column in candidates:
        base, model = column.rsplit("__", 1)
        stage = "derived" if base in DERIVED_SCORES else "phase3" if base.startswith("ros_") else "phase2"
        name = base[4:] if stage == "phase3" else base
        known = specific.get(column, definitions.get((stage, name)))
        if known is None and (base in DERIVED_SCORES or name == "replicate_composite"):
            known = {"direction": "higher_better", "group": "composite"}
        if known:
            found.add((stage, name))
        inferred = bool(known.get("inferred", False)) if known else True
        direction = known.get("direction", "unknown") if known else (
            infer_direction(table[column], evidence) if evidence is not None else "unknown")
        confirmed = bool(known.get("confirmed", not inferred)) if known else False
        rows.append({"column": column, "name": name, "stage": stage, "model": model,
                     "direction": direction, "group": known.get("group", "other") if known else ("consistency" if name.endswith("_std") else "other"),
                     "inferred": inferred, "confirmed": confirmed,
                     "missing_fraction": float(table[column].isna().mean()),
                     "constant": table[column].nunique(dropna=True) <= 1})
    warnings = [f"Registry entry not found: {stage}/{name}" for stage, name in definitions if (stage, name) not in found]
    return pd.DataFrame(rows, columns=["column", "name", "stage", "model", "direction", "group", "inferred", "confirmed", "missing_fraction", "constant"]), warnings


def direction_sign(direction: str) -> int:
    if direction not in DIRECTIONS - {"unknown"}:
        raise ValueError("A favorable score direction must be selected.")
    return 1 if direction == "higher_better" else -1