from __future__ import annotations

from dataclasses import replace
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .registry import GROUP_WEIGHTS, METRIC_GROUPS, SHARED_ELEC_COLS, direction_sign


OUTCOMES = {"binds": ["AUROC", "AUPRC", "Hit rate", "Enrichment"],
            "binding_ord": ["Spearman", "Kendall tau"], "log10_kd": ["Spearman"]}


def consolidation_scores(frame: pd.DataFrame, within_target: bool = True) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    group_scores = {}
    for group, metrics in METRIC_GROUPS.items():
        parts = []
        for column, sign in metrics.items():
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce") * sign
            ranked = values.groupby(frame.target).rank(pct=True, na_option="keep") if within_target else values.rank(pct=True, na_option="keep")
            result[f"pct_{column}"] = ranked * 100.0
            parts.append(result[f"pct_{column}"])
        if parts:
            group_scores[group] = pd.concat(parts, axis=1).mean(axis=1)
            result[f"score_{group}"] = group_scores[group]
    total_weight = sum(GROUP_WEIGHTS[group] for group in group_scores)
    result["composite"] = (sum(GROUP_WEIGHTS[group] * values for group, values in group_scores.items()) / total_weight
                           if total_weight else pd.Series(np.nan, index=frame.index))
    return result


def add_consolidation_scores(table: pd.DataFrame, models: list[str], replicate_level: bool = False) -> pd.DataFrame:
    result = table.copy()
    for model in models:
        ingredients = table[["sequence_id", "target"]].copy()
        for group, metrics in METRIC_GROUPS.items():
            for name in metrics:
                phase2, phase3 = f"{name}__{model}", f"ros_{name}__{model}"
                if name in SHARED_ELEC_COLS:
                    available = [column for column in [phase2, phase3] if column in table]
                    if available:
                        ingredients[name] = table[available].mean(axis=1)
                else:
                    preferred = [phase3, phase2] if group == "rosetta" else [phase2, phase3]
                    column = next((column for column in preferred if column in table), None)
                    if column:
                        ingredients[name] = table[column]
        observations = ingredients if replicate_level else ingredients.groupby(["sequence_id", "target"], sort=False).mean(numeric_only=True).reset_index()
        values = consolidation_scores(observations)
        for column in values:
            if replicate_level:
                result[f"{column}__{model}"] = values[column].to_numpy()
            else:
                lookup = pd.Series(values[column].to_numpy(), index=observations.sequence_id)
                result[f"{column}__{model}"] = table.sequence_id.map(lookup)
    return result


def sequence_observations(frame: pd.DataFrame, scores: list[str], outcome: str) -> pd.DataFrame:
    columns = list(dict.fromkeys(["sequence_id", outcome] + scores + (["target"] if "target" in frame else [])))
    data = frame[columns].copy()
    if outcome == "log10_kd" and "source" in frame:
        data = data.loc[frame["source"].eq("internal")]
    for column in scores + [outcome]:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    group = data.groupby("sequence_id", sort=False)
    functions = {column: "mean" for column in scores}
    functions[outcome] = lambda values: values.dropna().iloc[0] if values.dropna().nunique() == 1 else np.nan
    if "target" in data:
        functions["target"] = "first"
    return group.agg(functions).reset_index().dropna(subset=scores + [outcome])


def metric_value(labels: np.ndarray, scores: np.ndarray, metric: str, top_fraction: float = 0.1) -> float:
    if len(labels) < 2 or not np.isfinite(scores).all() or len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
        return np.nan
    if metric == "AUROC":
        return float(roc_auc_score(labels, scores))
    if metric == "AUPRC":
        return float(average_precision_score(labels, scores))
    if metric in {"Hit rate", "Enrichment"}:
        count = max(1, math.ceil(len(labels) * top_fraction))
        rate = float(labels[np.argsort(-scores, kind="stable")[:count]].mean())
        return rate if metric == "Hit rate" else rate / float(labels.mean())
    if metric == "Spearman":
        return float(spearmanr(labels, scores).statistic)
    if metric == "Kendall tau":
        return float(kendalltau(labels, scores).statistic)
    raise ValueError(f"Unknown metric {metric}")


def evaluate(frame: pd.DataFrame, score: str, outcome: str = "binds", metric: str = "AUROC",
             direction: str = "higher_better", n_boot: int = 200, seed: int = 42,
             top_fraction: float = 0.1) -> dict:
    if metric not in OUTCOMES[outcome]:
        raise ValueError(f"{metric} is not valid for {outcome}")
    data = sequence_observations(frame, [score], outcome)
    labels = data[outcome].to_numpy(dtype=float)
    if outcome == "log10_kd":
        labels = -labels
    scores = data[score].to_numpy(dtype=float) * direction_sign(direction)
    value = metric_value(labels, scores, metric, top_fraction)
    random = np.random.default_rng(seed)
    samples = []
    if len(data) >= 2 and np.isfinite(value):
        for _ in range(n_boot):
            indices = random.integers(0, len(data), len(data))
            estimate = metric_value(labels[indices], scores[indices], metric, top_fraction)
            if np.isfinite(estimate):
                samples.append(estimate)
    low, high = np.quantile(samples, [0.025, 0.975]) if samples else (np.nan, np.nan)
    positive = int((labels == 1).sum()) if outcome == "binds" else None
    return {"score": score, "metric": metric, "value": value, "ci_low": low, "ci_high": high,
            "n": len(data), "positives": positive, "negatives": len(data) - positive if positive is not None else None,
            "valid_bootstraps": len(samples), "small_n": len(data) < 30 or (positive is not None and min(positive, len(data) - positive) < 5)}


def exclusion_reason(frame: pd.DataFrame, entry: dict, include_unconfirmed: bool) -> str:
    if entry["column"] not in frame:
        return "column absent from the current table"
    if entry.get("constant", False):
        return "constant value"
    if entry["direction"] == "unknown":
        return "direction unknown"
    if not entry["confirmed"] and not include_unconfirmed:
        return "direction inferred, not confirmed"
    if frame[entry["column"]].notna().sum() < 2:
        return "fewer than two measured values"
    return ""


def score_eligibility(frame: pd.DataFrame, catalog: pd.DataFrame, include_unconfirmed=False) -> pd.DataFrame:
    rows = [{"score": entry["column"], "stage": entry.get("stage"), "model": entry.get("model"),
             "direction": entry["direction"], "confirmed": entry.get("confirmed", False),
             "missing_fraction": entry.get("missing_fraction"),
             "reason": exclusion_reason(frame, entry, include_unconfirmed)} for entry in catalog.to_dict("records")]
    table = pd.DataFrame(rows, columns=["score", "stage", "model", "direction", "confirmed", "missing_fraction", "reason"])
    table.insert(1, "eligible", table.reason.eq(""))
    return table


def rank_scores(frame: pd.DataFrame, catalog: pd.DataFrame, outcome="binds", metric="AUROC",
                include_unconfirmed=False, **kwargs) -> pd.DataFrame:
    rows = []
    for entry in catalog.to_dict("records"):
        if exclusion_reason(frame, entry, include_unconfirmed):
            continue
        row = evaluate(frame, entry["column"], outcome, metric, entry["direction"], **kwargs)
        rows.append({**entry, **row})
    return pd.DataFrame(rows).sort_values("value", ascending=False, na_position="last") if rows else pd.DataFrame()


def score_target_matrix(frame: pd.DataFrame, columns: list[str], directions: dict[str, str], outcome="binds",
                        metric="AUROC", minimum_sequences: int = 5, **kwargs) -> pd.DataFrame:
    groups = [("Overall", frame)]
    if "target" in frame:
        groups += [(str(target), subset) for target, subset in frame.groupby("target", sort=True)]
    rows = []
    for scope, subset in groups:
        for column in columns:
            if column not in subset:
                continue
            result = evaluate(subset, column, outcome, metric, directions.get(column, "higher_better"), **kwargs)
            if result["n"] < minimum_sequences:
                result["value"], result["ci_low"], result["ci_high"] = np.nan, np.nan, np.nan
            rows.append({"scope": scope, **result})
    return pd.DataFrame(rows)


def combination_gain(frame: pd.DataFrame, columns: list[str], directions: dict[str, str], outcome="binds",
                     metric="Hit rate", max_scores: int = 3, within_target: bool = True, **kwargs) -> pd.DataFrame:
    """Greedy forward search: best single score, then the best score to add at each step."""
    if len(columns) < 2:
        raise ValueError("Combination search requires at least two scores")
    work = frame.copy()
    chosen, remaining, rows, baseline = [], list(columns), [], np.nan
    for size in range(1, min(max_scores, len(columns)) + 1):
        best_column, best_result = None, None
        for candidate in remaining:
            trial = chosen + [candidate]
            work["combination_candidate"] = (pd.to_numeric(work[candidate], errors="coerce") * direction_sign(directions[candidate])
                                             if size == 1 else composite(work, dict.fromkeys(trial, 1.0), directions, "rank", within_target))
            result = evaluate(work, "combination_candidate", outcome, metric, "higher_better", **kwargs)
            if np.isfinite(result["value"]) and (best_result is None or result["value"] > best_result["value"]):
                best_column, best_result = candidate, result
        if best_result is None:
            break
        chosen.append(best_column)
        remaining.remove(best_column)
        baseline = best_result["value"] if size == 1 else baseline
        rows.append({"n_scores": size, "scores": " + ".join(chosen), "added": best_column, "metric": metric,
                     "value": best_result["value"], "ci_low": best_result["ci_low"], "ci_high": best_result["ci_high"],
                     "lift": best_result["value"] - baseline, "n": best_result["n"], "positives": best_result["positives"]})
    return pd.DataFrame(rows)


def composite(frame: pd.DataFrame, weights: dict[str, float], directions: dict[str, str],
              normalization: str = "rank", within_target: bool = True) -> pd.Series:
    if not weights or not any(abs(weight) > 0 for weight in weights.values()):
        raise ValueError("Composite requires at least one nonzero weight")
    scores = pd.DataFrame(index=frame.index)
    for column, weight in weights.items():
        if not weight:
            continue
        values = pd.to_numeric(frame[column], errors="coerce") * direction_sign(directions[column])
        group = values.groupby(frame["target"]) if within_target else None
        if normalization == "rank":
            scaled = group.rank(pct=True) if group is not None else values.rank(pct=True)
        elif normalization == "zscore":
            mean = group.transform("mean") if group is not None else values.mean()
            std = group.transform("std") if group is not None else values.std()
            scaled = (values - mean) / (std.replace(0, np.nan) if isinstance(std, pd.Series) else (std if std else np.nan))
        else:
            raise ValueError("Normalization must be rank or zscore")
        scores[column] = scaled * weight
    return scores.sum(axis=1, min_count=len(scores.columns)) / sum(abs(weight) for weight in weights.values())


def consensus(frame: pd.DataFrame, columns: list[str], directions: dict[str, str], mode="mean_rank",
              threshold: float = 0.8, minimum_models: int = 1, within_target: bool = True) -> pd.Series:
    models = [column.rsplit("__", 1)[-1] for column in columns]
    if len(set(models)) != len(models):
        raise ValueError("Consensus requires exactly one score per model")
    parts = pd.DataFrame(index=frame.index)
    for column in columns:
        normalization = "zscore" if mode == "mean_zscore" else "rank"
        parts[column] = composite(frame, {column: 1.0}, directions, normalization, within_target)
    if mode in {"mean_rank", "mean_zscore"}:
        result = parts.mean(axis=1)
    elif mode == "min":
        result = parts.min(axis=1)
    elif mode == "passing":
        result = parts.ge(threshold).sum(axis=1).astype(float)
    else:
        raise ValueError("Unknown consensus mode")
    return result.where(parts.notna().sum(axis=1) >= minimum_models)


def paired_auroc(frame: pd.DataFrame, score_a: str, score_b: str, directions: dict[str, str], n_boot=500, seed=42) -> dict:
    data = sequence_observations(frame, [score_a, score_b], "binds")
    labels = data.binds.to_numpy(dtype=float)
    scores_a = data[score_a].to_numpy(dtype=float) * direction_sign(directions[score_a])
    scores_b = data[score_b].to_numpy(dtype=float) * direction_sign(directions[score_b])
    delta = metric_value(labels, scores_a, "AUROC") - metric_value(labels, scores_b, "AUROC")
    random = np.random.default_rng(seed)
    values = []
    if np.isfinite(delta):
        for _ in range(n_boot):
            indices = random.integers(0, len(data), len(data))
            value = metric_value(labels[indices], scores_a[indices], "AUROC") - metric_value(labels[indices], scores_b[indices], "AUROC")
            if np.isfinite(value):
                values.append(value)
    bounds = np.quantile(values, [0.025, 0.975]) if values else [np.nan, np.nan]
    return {"delta_auroc": delta, "ci_low": bounds[0], "ci_high": bounds[1], "n": len(data),
            "bootstrap_p": min(1.0, 2 * min((np.sum(np.array(values) <= 0) + 1) / (len(values) + 1),
                                           (np.sum(np.array(values) >= 0) + 1) / (len(values) + 1))) if values else np.nan}


def fit_composite(frame: pd.DataFrame, columns: list[str], outcome="binds", seed=42) -> tuple[pd.Series, pd.DataFrame]:
    if outcome not in {"binds", "binding_ord", "log10_kd"}:
        raise ValueError("Unsupported fitted-composite outcome")
    observations = sequence_observations(frame, columns, outcome)
    if observations["target"].nunique() < 2:
        raise ValueError("Fitted composites require at least two labeled targets for grouped cross-validation")
    labels = observations[outcome].to_numpy(dtype=float)
    if outcome == "log10_kd":
        labels = -labels
    matrix = observations[columns].to_numpy(dtype=float)
    groups = observations["target"].to_numpy()
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.full(len(observations), np.nan)
    coefficient_rows = []

    def make_estimator():
        model = LogisticRegression(max_iter=2000, random_state=seed) if outcome == "binds" else Ridge(alpha=1.0)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    for fold, (train, test) in enumerate(splitter.split(matrix, labels, groups)):
        if outcome == "binds" and len(np.unique(labels[train])) < 2:
            raise ValueError("A target-held-out training fold has only one class; fitted composite is not estimable")
        estimator = make_estimator()
        estimator.fit(matrix[train], labels[train])
        oof[test] = estimator.predict_proba(matrix[test])[:, 1] if outcome == "binds" else estimator.predict(matrix[test])
        for column, coefficient in zip(columns, estimator[-1].coef_.ravel()):
            coefficient_rows.append({"fold": fold + 1, "score": column, "coefficient": float(coefficient),
                                     "held_out_targets": ", ".join(sorted(set(groups[test])))})
    production = make_estimator()
    production.fit(matrix, labels)
    all_values = frame[columns].apply(pd.to_numeric, errors="coerce")
    predicted = production.predict_proba(all_values.to_numpy())[:, 1] if outcome == "binds" else production.predict(all_values.to_numpy())
    result = pd.Series(predicted, index=frame.index)
    lookup = pd.Series(oof, index=observations.sequence_id)
    labeled = frame.sequence_id.isin(lookup.index)
    result.loc[labeled] = frame.loc[labeled, "sequence_id"].map(lookup)
    return result, pd.DataFrame(coefficient_rows)


def aggregation_sensitivity(config, score: str, outcome: str, metric: str, direction: str,
                            experiments: pd.DataFrame, **kwargs) -> pd.DataFrame:
    from .experiments import attach_experiments
    from .loader import load_predictions

    rows = []
    for mode in ["best", "mean", "worst"]:
        changed = replace(config, phase2=replace(config.phase2, mode=mode), phase3=replace(config.phase3, mode=mode))
        table = load_predictions(changed).table
        if score in table:
            rows.append({"aggregation": mode, **evaluate(attach_experiments(table, experiments), score, outcome, metric, direction, **kwargs)})
    return pd.DataFrame(rows)