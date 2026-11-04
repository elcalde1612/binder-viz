from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .loader import normalize_sequence, sequence_id
from .paths import input_files


CLASSES = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
LABELS = {"none": "none", "nan": "none", "no": "none", "negative": "none", "no binding": "none",
          "weak": "weak", "medium": "medium", "strong": "strong"}


def parse_kd(value: object, detection_limit_nM: float | None = None, default_unit: str = "nM") -> tuple[float, str]:
    if value is None or pd.isna(value) or str(value).strip().lower() in {"", "nan", "na", "n/a"}:
        return np.nan, "missing"
    text = str(value).strip().replace(",", "").replace("µ", "u").replace("μ", "u")
    if re.search(r"no\s+binding|not\s+detected", text, re.I):
        return (float(detection_limit_nM), ">") if detection_limit_nM else (np.nan, "no_binding_unknown_bound")
    match = re.fullmatch(r"\s*([<>]=?|~|=)?\s*([\d.]+(?:e[+-]?\d+)?)\s*(pM|nM|uM|mM|M)?\s*", text, re.I)
    if not match:
        return np.nan, "unparsed"
    qualifier, number, unit = match.groups()
    factors = {"pm": 0.001, "nm": 1, "um": 1e3, "mm": 1e6, "m": 1e9}
    try:
        numeric = float(number) * factors[(unit or default_unit).lower()]
    except ValueError:
        return np.nan, "unparsed"
    return (numeric, qualifier or "=") if numeric > 0 and np.isfinite(numeric) else (np.nan, "invalid")


@dataclass
class ExperimentalResult:
    table: pd.DataFrame
    warnings: list[str]
    unmatched: pd.DataFrame
    duplicates: pd.DataFrame


def load_experiments(files: list[dict], predictions: pd.DataFrame | None = None, label_mapping: dict | None = None,
                     duplicate_policy: str = "flag") -> ExperimentalResult:
    match_predictions = predictions is not None
    if predictions is None:
        predictions = pd.DataFrame(columns=["sequence_id", "target", "sequence"])
    mapping = {**LABELS, **{str(key).strip().lower(): value for key, value in (label_mapping or {}).items()}}
    warnings = []
    frames = []
    expanded = []
    for spec in files:
        try:
            expanded.extend({**spec, "path": str(path)} for path in input_files(spec["path"], (".csv",)))
        except (OSError, ValueError) as error:
            warnings.append(f"Could not discover experimental files in {spec.get('path')}: {error}")
    for spec in expanded:
        try:
            columns = dict(spec.get("columns", {}))
            identifiers = {"id", "name", "sequence_id", "design", "binder_id", "binder_name", "design_id", "design_name",
                           "sequence_name", "experimental_id", "experimental_name"}
            identifiers.update(original for canonical, original in columns.items() if canonical in identifiers and original)
            frame = pd.read_csv(Path(spec["path"]).expanduser(), keep_default_na=False,
                                dtype=dict.fromkeys(identifiers, "string"))
            for canonical, original in [("binding_class", "binding_strength"), ("kd_nM", "KD (M)")]:
                if canonical not in frame and canonical not in columns and original in frame:
                    columns[canonical] = original
            kd_unit = spec.get("kd_unit", "M" if columns.get("kd_nM") == "KD (M)" else "nM")
            if kd_unit not in {"pM", "nM", "uM", "mM", "M"}:
                raise ValueError("Unsupported Kd unit")
            frame = frame.rename(columns={original: canonical for canonical, original in columns.items() if original})
            for destination, candidates in [("experimental_id", ["id", "sequence_id", "binder_id", "design_id"]),
                                             ("experimental_name", ["name", "design", "binder_name", "design_name", "sequence_name"])]:
                values = frame.get(destination, pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string").str.strip().replace("", pd.NA)
                for candidate in candidates:
                    if candidate in frame:
                        values = values.fillna(frame[candidate].astype("string").str.strip().replace("", pd.NA))
                frame[destination] = values
            source = spec.get("source", "auto")
            if source not in {"auto", "internal", "public"}:
                raise ValueError("Experimental source must be auto, internal or public")
            if source == "auto":
                origins = frame.get("data_origin", frame.get("source", pd.Series("internal", index=frame.index))).astype(str).str.strip().str.lower()
                sources = origins.map({"proteinbase": "public", "public": "public", "internal": "internal"})
                if sources.isna().any():
                    warnings.append(f"{spec['path']}: unknown sources treated as public: {sorted(origins[sources.isna()].unique())}")
                frame["source"] = sources.fillna("public")
            else:
                frame["source"] = source
            if "sequence" in frame:
                if "target" not in frame:
                    targets = predictions["target"].unique()
                    default = spec.get("target") or (targets[0] if len(targets) == 1 else None)
                    if not default:
                        raise ValueError("Experimental file requires a target column or a configured target")
                    frame["target"] = default
                frame["sequence"] = frame["sequence"].map(normalize_sequence)
                frame["target"] = frame["target"].astype(str).str.strip()
                valid = frame["sequence"].ne("") & frame["target"].ne("")
                if not valid.all():
                    warnings.append(f"{spec['path']}: {(~valid).sum()} rows missing target/sequence excluded")
                frame = frame[valid].copy()
                frame["sequence_id"] = [sequence_id(target, seq) for target, seq in zip(frame["target"], frame["sequence"])]
            elif "sequence_id" not in frame:
                raise ValueError("Experimental files require sequence_id or sequence")
            labels = frame.get("binding_class", pd.Series("", index=frame.index)).astype(str).str.strip().str.lower()
            frame["binding_class"] = labels.map(mapping)
            unknown = labels.ne("") & frame["binding_class"].isna()
            if unknown.any():
                warnings.append(f"{spec['path']}: {unknown.sum()} unknown class labels: {sorted(labels[unknown].unique())}")
            if not frame["binding_class"].dropna().isin(CLASSES).all():
                raise ValueError("Class mappings must resolve to none, weak, medium or strong")
            binary = frame.get("binding", pd.Series("", index=frame.index)).astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}).astype("boolean")
            frame.loc[frame.binding_class.isna() & binary.eq(False).fillna(False), "binding_class"] = "none"
            frame["binding_ord"] = frame["binding_class"].map(CLASSES).astype("Float64")
            frame["binds"] = frame["binding_ord"].gt(0).astype("boolean").fillna(binary)
            kd = frame.get("kd_nM", pd.Series(None, index=frame.index, dtype=object)).map(lambda value: parse_kd(value, spec.get("detection_limit_nM"), kd_unit))
            frame["kd_nM"] = [item[0] for item in kd]
            provided = frame.get("kd_qualifier", pd.Series("", index=frame.index)).astype(str).str.strip()
            frame["kd_qualifier"] = [item[1] for item in kd]
            frame.loc[provided.isin(["<", ">", "<=", ">=", "~", "="]), "kd_qualifier"] = provided
            frame["log10_kd"] = np.log10(frame["kd_nM"])
            public = frame.source.eq("public")
            frame.loc[public, ["kd_nM", "log10_kd"]] = np.nan
            frame.loc[public, "kd_qualifier"] = "public_not_applicable"
            fsec = [column for column in frame if column.startswith("fsec_")]
            frame.loc[public, fsec] = np.nan
            if public.all():
                frame = frame.drop(columns=fsec)
            unbounded = frame["kd_qualifier"].isin(["no_binding_unknown_bound", "unparsed", "invalid"])
            if unbounded.any():
                warnings.append(f"{spec['path']}: {unbounded.sum()} Kd values lack a usable bound; retained but excluded from regression")
            allowed = ["sequence_id", "experimental_id", "experimental_name", "target", "sequence", "source", "binding_class", "binding_ord", "binds", "kd_nM", "log10_kd", "kd_qualifier"]
            allowed += [column for column in frame if column.startswith("fsec_")]
            frames.append(frame[[column for column in allowed if column in frame]].assign(experiment_file=str(spec["path"])))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            warnings.append(f"Could not load experimental file {spec.get('path')}: {error}")
    empty = pd.DataFrame(columns=["sequence_id", "source", "binding_class", "binding_ord", "binds", "kd_nM", "log10_kd", "kd_qualifier"])
    table = pd.concat(frames, ignore_index=True) if frames else empty
    reference = predictions.drop_duplicates("sequence_id").set_index("sequence_id")
    for column in ("target", "sequence"):
        resolved = table.sequence_id.map(reference.get(column, pd.Series(dtype=str)))
        table[column] = table[column].replace("", pd.NA).fillna(resolved) if column in table else resolved
    table["sequence"] = table.sequence.map(normalize_sequence)
    keys = ["sequence_id", "source"]
    duplicates = table[table.duplicated(keys, keep=False)].copy()
    if not duplicates.empty:
        if duplicate_policy == "error":
            raise ValueError("Multiple experimental measurements per sequence/source")
        warnings.append(f"{len(duplicates)} experimental rows share sequence/source; policy={duplicate_policy}")
    if not table.empty:
        table["n_measurements"] = table.groupby(keys)["sequence_id"].transform("size")
        if duplicate_policy == "aggregate":
            rows = []
            for _, group in table.groupby(keys, sort=False):
                row = group.iloc[0].copy()
                ordinal = group["binding_ord"].dropna().median()
                row["binding_ord"] = float(np.floor(ordinal + 0.5)) if pd.notna(ordinal) else np.nan
                row["binding_class"] = {value: key for key, value in CLASSES.items()}.get(row["binding_ord"], None)
                row["binds"] = row["binding_ord"] > 0 if pd.notna(row["binding_ord"]) else group["binds"].dropna().max()
                qualifiers = group.loc[group["kd_nM"].notna(), "kd_qualifier"].unique()
                if len(qualifiers) > 1:
                    row["kd_nM"], row["log10_kd"], row["kd_qualifier"] = np.nan, np.nan, "mixed_bounds"
                    warnings.append("Mixed exact/censored Kd replicates cannot be averaged; kinetics excluded for that sequence")
                else:
                    row["kd_nM"] = group["kd_nM"].median()
                    row["log10_kd"] = np.log10(row["kd_nM"]) if pd.notna(row["kd_nM"]) else np.nan
                rows.append(row)
            table = pd.DataFrame(rows).reset_index(drop=True)
    unmatched = table[~table["sequence_id"].isin(predictions["sequence_id"])].copy()
    if match_predictions and not unmatched.empty:
        warnings.append(f"{len(unmatched)} experimental rows do not match predictions")
    return ExperimentalResult(table, warnings, unmatched, duplicates)


def attach_experiments(predictions: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in experiments if column not in {"target", "sequence"}]
    return predictions.merge(experiments[columns], on="sequence_id", how="left", validate="many_to_many")


def sequence_labels(frame: pd.DataFrame) -> pd.Series:
    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in ["experimental_name", "experimental_id", "sequence", "sequence_id"]:
        if column in frame:
            labels = labels.fillna(frame[column].astype("string").str.strip().replace("", pd.NA))
    return labels.fillna("Unnamed sequence")