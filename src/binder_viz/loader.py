from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from .config import Aggregation, SessionConfig


PHASE_PATTERN = re.compile(r"^.+_phase(?P<phase>[23])(?:_stats)?\.csv$", re.I)
RELAX_SUFFIX = re.compile(r"_\d{4}$")
BOOKKEEPING = {
    "sequence", "pdbname", "description", "record_id", "method", "phase", "stage",
    "global_iteration", "epoch", "pass_filter", "parent_backbone", "origin", "status",
    "seed", "sample", "replicate", "replicate_id", "target", "model", "name", "design",
}
KEYS = ["target", "model", "sequence"]


def normalize_name(value: object, relaxed: bool = True) -> str:
    if pd.isna(value):
        return ""
    name = re.sub(r"\.(?:pdb|cif)(?:\.gz)?$", "", str(value).strip(), flags=re.I)
    return RELAX_SUFFIX.sub("", name) if relaxed else name


def normalize_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value)).upper()


def sequence_id(target: str, sequence: str) -> str:
    return sha256(json.dumps([str(target).strip(), normalize_sequence(sequence)]).encode()).hexdigest()[:20]


def discover_files(folder: str | Path) -> list[tuple[Path, int]]:
    root = Path(folder).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.csv"))
    return [(path, int(match["phase"])) for path in paths
            if (match := PHASE_PATTERN.match(path.name))]


def discover_run_folders(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser()
    files = discover_files(root)
    return sorted({path.parent.parent if path.parent.name == "traj_stats" else path.parent for path, _ in files})


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if column not in BOOKKEEPING and not column.startswith("_")
            and not column.startswith("Unnamed:") and pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])]


def _read_stage(path: Path, phase: int, target: str, model: str, warnings: list[str], nrows: int | None = None) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, nrows=nrows)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        warnings.append(f"Could not read {path}: {error}")
        return pd.DataFrame()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        warnings.append(f"{path}: duplicate column names; file skipped")
        return pd.DataFrame()
    frame["target"], frame["model"] = target, model
    frame["_file"] = str(path)
    frame["sequence"] = frame.get("sequence", pd.Series("", index=frame.index)).map(normalize_sequence)
    name_col = "pdbname" if phase == 2 else "description"
    if name_col not in frame:
        name_col = "description" if "description" in frame else "pdbname"
    names = frame.get(name_col, pd.Series("", index=frame.index))
    frame["_name"] = names.map(lambda value: normalize_name(value, relaxed=phase == 3))
    parsed = frame["_name"].str.extract(r"(?:^|_)(?:seed|sample|model|replicate|rep)[_-]?(\d+)(?:_|$)", expand=False)
    explicit = next((column for column in ("replicate_id", "replicate", "seed", "sample") if column in frame), None)
    frame["_rep"] = frame[explicit].astype("string") if explicit else parsed.astype("string")
    frame["_duplicate_key"] = frame["_name"].where(frame["_name"].ne(""), frame["_rep"])
    missing_key = frame["_duplicate_key"].isna() | frame["_duplicate_key"].eq("")
    if missing_key.any():
        frame.loc[missing_key, "_duplicate_key"] = [f"{path}:{index}" for index in frame.index[missing_key]]
        warnings.append(f"{path.name}: {missing_key.sum()} rows lack replicate identifiers; duplicate detection is limited")
    for column in frame.columns:
        if column in BOOKKEEPING or column.startswith("_") or column.startswith("Unnamed:"):
            continue
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        if converted.notna().any() or pd.api.types.is_numeric_dtype(original):
            lost = int((original.notna() & converted.isna()).sum())
            if lost:
                warnings.append(f"{path.name}: {column}: {lost} nonnumeric values treated as missing")
            frame[column] = converted.replace([np.inf, -np.inf], np.nan)
        elif original.notna().any():
            warnings.append(f"{path.name}: {column}: nonnumeric column excluded from scores")
    if "ipsae" not in frame and "ipsae_binder" in frame:
        frame["ipsae"] = frame["ipsae_binder"]
        warnings.append(f"{path.name}: used ipsae_binder as ipsae")
    return frame


def replicate_selector_columns(folders) -> tuple[dict[str, list[str]], list[str]]:
    from .registry import METRIC_GROUPS

    warnings = []
    options = {"phase2": set(), "phase3": set()}
    for folder in folders:
        for path, phase in discover_files(folder.path):
            frame = _read_stage(path, phase, folder.target, folder.model, warnings, nrows=256)
            options[f"phase{phase}"].update(numeric_columns(frame))
    ingredients = {name for metrics in METRIC_GROUPS.values() for name in metrics}
    for columns in options.values():
        if columns.intersection(ingredients):
            columns.add("replicate_composite")
    return {stage: sorted(columns) for stage, columns in options.items()}, list(dict.fromkeys(warnings))


def aggregate_stage(frame: pd.DataFrame, settings: Aggregation, warnings: list[str] | None = None) -> pd.DataFrame:
    warnings = warnings if warnings is not None else []
    if frame.empty:
        return frame.copy()
    frame = frame.reset_index(drop=True)
    if settings.selector == "replicate_composite":
        from .analysis import consolidation_scores

        frame[settings.selector] = np.nan
        for _, group in frame.groupby("model", sort=False):
            frame.loc[group.index, settings.selector] = consolidation_scores(group).composite
    elif settings.composite:
        from .analysis import composite

        weights = settings.composite["weights"]
        missing = [column for column, weight in weights.items() if weight and column not in frame]
        for column in missing:
            warnings.append(f"Composite selector {settings.selector}: ingredient {column!r} missing")
        ingredients = frame.copy()
        for column in missing:
            ingredients[column] = np.nan
        frame[settings.selector] = np.nan
        for _, group in ingredients.groupby("model", sort=False):
            frame.loc[group.index, settings.selector] = composite(
                group, weights, settings.composite["directions"], settings.composite.get("normalization", "rank"),
                settings.composite.get("within_target", True))
    scores = numeric_columns(frame)
    group = frame.groupby(KEYS, sort=False, dropna=False)
    metadata = [column for column in ["_name", "_rep", "_duplicate_key", "_file"] if column in frame]
    if settings.mode == "none":
        out = frame[KEYS + metadata + scores].copy()
        out["n_replicates"] = group["sequence"].transform("size")
    else:
        if settings.mode in {"best", "worst"}:
            if settings.selector not in scores:
                warnings.append(f"Selector {settings.selector!r} missing; retained first row per sequence")
                chosen = group.head(1)
            else:
                ascending = settings.direction == "lower_better"
                if settings.mode == "worst":
                    ascending = not ascending
                chosen = frame.sort_values(settings.selector, ascending=ascending, na_position="last", kind="stable").groupby(KEYS, sort=False).head(1)
                if group[settings.selector].count().eq(0).any():
                    warnings.append(f"{settings.selector}: all-missing selector in some groups; retained first row")
            out = chosen.set_index(KEYS)[metadata + scores].copy()
        else:
            out = getattr(group[scores], settings.mode)()
        for column, mode in settings.overrides.items():
            if column not in scores:
                warnings.append(f"Aggregation override column {column!r} not found")
                continue
            if mode in {"mean", "median"}:
                out[column] = getattr(group[column], mode)()
            else:
                favorable_max = settings.direction == "higher_better"
                use_max = favorable_max if mode == "best" else not favorable_max
                out[column] = group[column].max() if use_max else group[column].min()
        out["n_replicates"] = group.size()
        out["_files"] = group["_file"].agg(lambda values: " | ".join(sorted(set(values)))) if "_file" in frame else ""
        out = out.reset_index()
    if settings.std and scores:
        spread = group[scores].std().rename(columns=lambda column: f"{column}_std").reset_index()
        out = out.merge(spread, on=KEYS, how="left", validate="many_to_one")
    return out


@dataclass
class LoadResult:
    table: pd.DataFrame
    report: pd.DataFrame
    replicate_counts: pd.DataFrame
    warnings: list[str]
    cache_key: str


def content_key(config: SessionConfig) -> str:
    digest = sha256(b"binder-viz-loader-v3")
    digest.update(config.to_yaml().encode())
    for folder in config.folders:
        for path, phase in discover_files(folder.path):
            if phase == 3 and not config.include_phase3:
                continue
            digest.update(str(path.resolve()).encode())
            try:
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as error:
                digest.update(str(error).encode())
    return digest.hexdigest()


def load_predictions(config: SessionConfig, cache_dir: str | Path | None = None) -> LoadResult:
    key = content_key(config)
    cache = Path(cache_dir) / key if cache_dir else None
    if cache and (cache / "metadata.json").exists():
        try:
            metadata = json.loads((cache / "metadata.json").read_text())
            return LoadResult(pd.read_parquet(cache / "table.parquet"), pd.DataFrame(metadata["report"]),
                              pd.read_parquet(cache / "replicates.parquet"), metadata["warnings"], key)
        except (OSError, ValueError):
            pass
    warnings: list[str] = []
    stages: dict[int, list[pd.DataFrame]] = {2: [], 3: []}
    for folder in config.folders:
        files = discover_files(folder.path)
        if not files:
            warnings.append(f"No phase CSVs found in {folder.path}")
        for path, phase in files:
            if phase == 3 and not config.include_phase3:
                continue
            frame = _read_stage(path, phase, folder.target, folder.model, warnings)
            if not frame.empty:
                stages[phase].append(frame)
    raw = {stage: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for stage, frames in stages.items()}
    if not raw[2].empty and not raw[3].empty:
        reference = raw[2].loc[raw[2]["_name"].ne("") & raw[2]["sequence"].ne(""), ["target", "model", "_name", "sequence"]].drop_duplicates()
        unique = ~reference.duplicated(["target", "model", "_name"], keep=False)
        if not unique.all():
            warnings.append("Ambiguous phase2 names map to multiple sequences; those names cannot recover phase3 sequences")
        mapping = reference[unique].rename(columns={"sequence": "_resolved_sequence"})
        raw[3] = raw[3].merge(mapping, on=["target", "model", "_name"], how="left", validate="many_to_one")
        conflict = raw[3]["sequence"].ne("") & raw[3]["_resolved_sequence"].notna() & raw[3]["sequence"].ne(raw[3]["_resolved_sequence"])
        if conflict.any():
            warnings.append(f"{conflict.sum()} phase3 sequence/name conflicts; explicit sequences retained")
        raw[3]["sequence"] = raw[3]["sequence"].mask(raw[3]["sequence"].eq(""), raw[3]["_resolved_sequence"]).fillna("")
        raw[3] = raw[3].drop(columns="_resolved_sequence")
    duplicate_counts: dict[int, dict] = {}
    counts = []
    for stage, frame in raw.items():
        if frame.empty:
            continue
        invalid = frame["sequence"].eq("")
        if invalid.any():
            warnings.append(f"phase{stage}: {invalid.sum()} rows have no recoverable sequence and are excluded")
        frame = frame[~invalid].copy()
        duplicated = frame.duplicated(KEYS + ["_duplicate_key"], keep=False)
        duplicate_counts[stage] = frame[duplicated].groupby(["target", "model"]).size().to_dict()
        if duplicated.any():
            message = f"phase{stage}: {duplicated.sum()} rows share duplicate (target, model, sequence, replicate) keys"
            if config.duplicates == "error":
                raise ValueError(message)
            warnings.append(f"{message}; keeping {config.duplicates}")
            frame = frame.drop_duplicates(KEYS + ["_duplicate_key"], keep=config.duplicates)
        raw[stage] = frame
        counts.append(frame.groupby(KEYS).size().rename("n_replicates").reset_index().assign(stage=f"phase{stage}"))
    replicate_counts = pd.concat(counts, ignore_index=True) if counts else pd.DataFrame(columns=KEYS + ["n_replicates", "stage"])
    both_none = config.include_phase3 and config.phase2.mode == config.phase3.mode == "none"
    replicate_mode = both_none or (not config.include_phase3 and config.phase2.mode == "none")
    if config.include_phase3 and ((config.phase2.mode == "none") != (config.phase3.mode == "none")):
        warnings.append("Replicate joins require none for both stages; using mean for the unaggregated stage")
    if replicate_mode:
        active = [frame for frame in raw.values() if not frame.empty]
        if any(frame["_rep"].isna().any() or frame.duplicated(KEYS + ["_rep"]).any() for frame in active):
            warnings.append("Replicate IDs cannot be parsed uniquely; falling back to sequence-level mean aggregation")
            replicate_mode = False
    aggregated = {}
    for stage, frame in raw.items():
        settings = config.phase2 if stage == 2 else config.phase3
        if settings.mode == "none" and not replicate_mode:
            settings = replace(settings, mode="mean")
        aggregated[stage] = aggregate_stage(frame, settings, warnings)
    models = sorted({folder.model for folder in config.folders})
    joined_models = []
    reports = []
    join_keys = KEYS + (["_rep"] if replicate_mode else [])
    for model in models:
        sides = []
        for stage, frame in aggregated.items():
            if frame.empty:
                continue
            side = frame[frame["model"].eq(model)].copy()
            if side.empty:
                continue
            score_cols = [column for column in numeric_columns(side) if column != "n_replicates"]
            prefix = "" if stage == 2 else "ros_"
            rename = {column: f"{prefix}{column}__{model}" for column in score_cols}
            rename["n_replicates"] = f"n_rep_p{stage}__{model}"
            provenance = "_files" if "_files" in side else "_file"
            extras = [provenance] if provenance in side else []
            rename.update({column: f"files_p{stage}__{model}" for column in extras})
            sides.append(side[join_keys + score_cols + ["n_replicates"] + extras].rename(columns=rename))
        if not sides:
            continue
        joined = sides[0]
        if len(sides) == 2:
            joined = joined.merge(sides[1], on=join_keys, how="outer", validate="one_to_one")
        for target, part in joined.groupby("target"):
            left = part.get(f"n_rep_p2__{model}", pd.Series(np.nan, index=part.index)).notna()
            right = part.get(f"n_rep_p3__{model}", pd.Series(np.nan, index=part.index)).notna()
            reports.append({"target": target, "model": model, "matched": int((left & right).sum()),
                            "unmatched_phase2": int((left & ~right).sum()), "unmatched_phase3": int((right & ~left).sum()),
                            "duplicate_rows_p2": duplicate_counts.get(2, {}).get((target, model), 0),
                            "duplicate_rows_p3": duplicate_counts.get(3, {}).get((target, model), 0),
                            "join_strategy": "replicate" if replicate_mode else "sequence"})
        joined_models.append(joined.drop(columns="model"))
    wide_keys = ["target", "sequence"] + (["_rep"] if replicate_mode else [])
    table = pd.DataFrame(columns=wide_keys)
    for frame in joined_models:
        table = frame if table.empty else table.merge(frame, on=wide_keys, how="outer", validate="one_to_one")
    if "_rep" in table:
        table = table.rename(columns={"_rep": "replicate_id"})
    table.insert(0, "sequence_id", [sequence_id(target, sequence) for target, sequence in zip(table["target"], table["sequence"])])
    for stage in (2, 3):
        columns = [column for column in table if column.startswith(f"n_rep_p{stage}__")]
        table[f"n_rep_p{stage}"] = table[columns].fillna(0).sum(axis=1).astype(int) if columns else 0
    from .analysis import add_consolidation_scores

    table = add_consolidation_scores(table, models)
    for column in [column for column in table if "__" in column and not column.startswith(("n_rep_", "files_"))]:
        if table[column].isna().mean() > 0.5:
            warnings.append(f"{column}: more than 50% missing values")
        if table[column].nunique(dropna=True) <= 1:
            warnings.append(f"{column}: constant or entirely missing")
    result = LoadResult(table, pd.DataFrame(reports), replicate_counts, list(dict.fromkeys(warnings)), key)
    if cache:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            table.to_parquet(cache / "table.parquet", index=False)
            replicate_counts.to_parquet(cache / "replicates.parquet", index=False)
            (cache / "metadata.json").write_text(json.dumps({"report": reports, "warnings": result.warnings}))
        except (OSError, ValueError) as error:
            result.warnings.append(f"Parquet cache unavailable: {error}")
    return result