from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import zipfile

from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np
import pandas as pd

from .registry import direction_sign


def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"
    return aligner


def pair_identity(aligner: PairwiseAligner, first: str, second: str) -> float:
    alignment = aligner.align(first, second)[0]
    counts = alignment.counts()
    denominator = counts.identities + counts.mismatches + counts.gaps
    return counts.identities / denominator if denominator else 0.0


COMBINATION_RULES = ["mean_rank", "min_rank", "intersection", "union"]
COMBINED_SCORE = "ranking_score__combined"


def score_ranks(table: pd.DataFrame, columns: list[str], directions: dict[str, str], within_target: bool = True) -> pd.DataFrame:
    ranks = pd.DataFrame(index=table.index)
    for column in columns:
        if column not in table:
            raise ValueError(f"Ranking score {column!r} is not available")
        if directions.get(column, "unknown") not in {"higher_better", "lower_better"}:
            raise ValueError(f"Ranking score {column!r} has no favorable direction")
        values = pd.to_numeric(table[column], errors="coerce") * direction_sign(directions[column])
        ranks[column] = values.groupby(table["target"]).rank(pct=True) if within_target else values.rank(pct=True)
    return ranks


def combine_scores(table: pd.DataFrame, weights: dict[str, float], directions: dict[str, str],
                   rule: str = "mean_rank", percentile: float = 0.8, within_target: bool = True) -> pd.Series:
    """Combine several scores into one higher-is-better ranking of within-target percentile ranks."""
    if rule not in COMBINATION_RULES:
        raise ValueError(f"Unknown combination rule {rule!r}")
    columns = [column for column, weight in weights.items() if weight]
    if not columns:
        raise ValueError("Combined ranking requires at least one nonzero weight")
    ranks = score_ranks(table, columns, directions, within_target)
    complete = ranks.notna().all(axis=1)
    if rule == "min_rank":
        combined = ranks.min(axis=1).where(complete)
    else:
        total = sum(abs(weights[column]) for column in columns)
        combined = sum(ranks[column] * weights[column] for column in columns) / total
    if rule == "intersection":
        combined = combined.where(ranks.ge(percentile).all(axis=1) & complete)
    elif rule == "union":
        combined = combined.where(ranks.ge(percentile).any(axis=1))
    return combined


def candidate_pool(table: pd.DataFrame, score: str, direction: str, thresholds: list[dict] | None = None,
                   tested: set[str] | None = None, agreement_columns: list[str] | None = None,
                   directions: dict[str, str] | None = None, minimum_models: int = 0,
                   agreement_percentile: float = 0.8, positive_targets: list[str] | None = None,
                   negative_selection: list[dict] | None = None, off_target_experiments: pd.DataFrame | None = None,
                   exclude_missing_off_targets: bool = True, off_target_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    pool = table.copy()
    reference = table if off_target_predictions is None else off_target_predictions
    negative_targets = {spec["target"] for spec in negative_selection or []}
    if negative_targets.intersection(positive_targets or []):
        raise ValueError("Positive and negative targets must not overlap")
    if positive_targets:
        pool = pool[pool.target.isin(positive_targets)]
    pool = pool[~pool.target.isin(negative_targets)]
    for spec in negative_selection or []:
        target = spec["target"]
        off_target = reference[reference.target.eq(target)]
        evidence = pd.Series(False, index=pool.index)
        rejected = pd.Series(False, index=pool.index)
        column = spec.get("score")
        if column:
            if column not in reference:
                raise ValueError(f"Off-target score {column!r} is not available")
            off_direction = spec.get("direction", (directions or {}).get(column, "unknown"))
            if off_direction not in {"higher_better", "lower_better"}:
                raise ValueError("Off-target score direction must be specified")
            threshold = spec.get("threshold")
            if threshold is None or not np.isfinite(float(threshold)):
                raise ValueError("Off-target binding threshold must be a finite number")
            grouped = off_target.groupby("sequence")[column]
            worst = grouped.max() if off_direction == "higher_better" else grouped.min()
            values = pool.sequence.map(worst)
            pool[f"off_target__{target}__{column}"] = values
            evidence |= values.notna()
            rejected |= values.ge(float(threshold)) if off_direction == "higher_better" else values.le(float(threshold))
        if off_target_experiments is not None and not off_target_experiments.empty:
            measurements = off_target_experiments[off_target_experiments.target.eq(target)].dropna(subset=["binds"])
            measured = pool.sequence.map(measurements.groupby("sequence").binds.max())
            pool[f"off_target__{target}__binds"] = measured
            evidence |= measured.notna()
            rejected |= measured.eq(True).fillna(False)
        pool[f"off_target__{target}__status"] = np.where(evidence, "passes", "missing evidence")
        pool = pool[~rejected & (evidence if exclude_missing_off_targets else True)]
    if negative_targets:
        pool["negative_targets_checked"] = ", ".join(sorted(negative_targets))
    if tested:
        pool = pool[~pool.sequence_id.isin(tested)]
    for spec in thresholds or []:
        column = spec["score"]
        if column not in pool:
            raise ValueError(f"Filter score {column!r} is not available")
        scope = pool.target.eq(spec["target"]) if spec.get("target") else pd.Series(True, index=pool.index)
        acceptable = pd.Series(True, index=pool.index)
        if spec.get("min") is not None and pd.notna(spec["min"]):
            acceptable &= pool[column].ge(float(spec["min"]))
        if spec.get("max") is not None and pd.notna(spec["max"]):
            acceptable &= pool[column].le(float(spec["max"]))
        pool = pool[~scope | acceptable]
    pool = pool.sort_values(score, ascending=direction == "lower_better", kind="stable").dropna(subset=[score]).drop_duplicates("sequence_id")
    rank_columns = []
    for column in agreement_columns or []:
        model = column.rsplit("__", 1)[-1]
        rank_column = f"rank__{model}"
        if rank_column in rank_columns:
            raise ValueError("Model agreement requires one score per model")
        reference = table.drop_duplicates("sequence_id").copy()
        oriented = reference[column] * direction_sign((directions or {})[column])
        ranks = oriented.groupby(reference.target).rank(pct=True)
        lookup = pd.Series(ranks.to_numpy(), index=reference.sequence_id)
        pool[rank_column] = pool.sequence_id.map(lookup)
        rank_columns.append(rank_column)
    if rank_columns:
        ranks = pool[rank_columns]
        pool["models_passing"] = ranks.ge(agreement_percentile).sum(axis=1)
        pool["model_rank_gap"] = ranks.max(axis=1) - ranks.min(axis=1)
        pool = pool[pool.models_passing.ge(minimum_models)]
    elif minimum_models:
        raise ValueError("Select model scores before applying an agreement requirement")
    flags = pd.Series("", index=pool.index)
    flags += np.where(pool.get("n_rep_p2", pd.Series(0, index=pool.index)).lt(2), "few phase2 replicates; ", "")
    if "model_rank_gap" in pool:
        flags += np.where(pool.model_rank_gap.gt(0.5), "models disagree; ", "")
    status_columns = [column for column in pool if column.startswith("off_target__") and column.endswith("__status")]
    if status_columns:
        flags += np.where(pool[status_columns].eq("missing evidence").any(axis=1), "missing off-target evidence; ", "")
    std_cols = [column for column in pool if "_std__" in column]
    high_variance = pd.Series(False, index=pool.index)
    for column in std_cols:
        reference = table[column].dropna()
        if len(reference) > 1:
            high_variance |= pool[column].gt(reference.quantile(0.9))
    flags += np.where(high_variance, "high replicate variance; ", "")
    pool["flags"] = flags.str.rstrip("; ")
    pool["selection_rank"] = np.arange(1, len(pool) + 1)
    return pool.reset_index(drop=True)


def select_candidates(pool: pd.DataFrame, count: int, identity: float = 0.8, per_cluster: int = 1,
                      per_target: int = 100, score: str = "", cluster: bool = True,
                      target_caps: dict[str, int] | None = None) -> pd.DataFrame:
    if not 0 < identity <= 1 or min(count, per_cluster, per_target) < 1:
        raise ValueError("Identity must be in (0, 1] and selection limits must be positive")
    aligner = make_aligner()
    representatives: list[tuple[str, str, int]] = []
    cluster_counts, target_counts = Counter(), Counter()
    rows = []
    for _, row in pool.iterrows():
        target = row.target
        if target_counts[target] >= (target_caps or {}).get(target, per_target):
            continue
        sequence = row.sequence
        unknown = set(sequence) - set(aligner.substitution_matrix.alphabet)
        if unknown:
            raise ValueError(f"Unsupported amino acid symbols for alignment: {sorted(unknown)}")
        label = None
        if cluster:
            for other_target, representative, existing_label in representatives:
                if target == other_target and pair_identity(aligner, sequence, representative) >= identity:
                    label = existing_label
                    break
        if label is None:
            label = len(representatives) + 1
            representatives.append((target, sequence, label))
        if cluster_counts[label] >= per_cluster:
            continue
        selected = row.to_dict()
        selected.update(cluster_id=label, selected=True,
                        selection_rationale=f"rank={row.get('selection_rank')}; score={score}; identity_cap={identity}; cluster={label}")
        if row.get("negative_targets_checked"):
            selected["selection_rationale"] += f"; negative_targets={row['negative_targets_checked']}"
        rows.append(selected)
        cluster_counts[label] += 1
        target_counts[target] += 1
        if len(rows) == count:
            break
    return pd.DataFrame(rows) if rows else pool.iloc[:0].assign(cluster_id=pd.Series(dtype=int), selected=pd.Series(dtype=bool), selection_rationale=pd.Series(dtype=str))


def export_selection(table: pd.DataFrame, config: dict) -> dict[str, bytes]:
    config_json = json.dumps(config, sort_keys=True)
    config_hash = sha256(config_json.encode()).hexdigest()[:16]
    exported = table.copy()
    exported["session_config_json"] = config_json
    csv = exported.to_csv(index=False).encode()
    lines = [f"; binder-viz session_config_json={config_json}"]
    for row in table.to_dict("records"):
        target = str(row["target"]).replace("\n", " ").replace("\r", " ")
        rationale = str(row.get("selection_rationale", "manual")).replace("\n", " ")
        lines += [f">{row['sequence_id']} target={target} config={config_hash} {rationale}", row["sequence"]]
    fasta = ("\n".join(lines) + "\n").encode()
    import yaml

    session = yaml.safe_dump(config, sort_keys=False).encode()
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("selected.csv", csv)
        handle.writestr("selected.fasta", fasta)
        handle.writestr("session.yaml", session)
    return {"csv": csv, "fasta": fasta, "yaml": session, "zip": archive.getvalue()}