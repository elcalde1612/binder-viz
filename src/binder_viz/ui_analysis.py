from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .analysis import (OUTCOMES, aggregation_sensitivity, combination_gain, composite, consensus, evaluate,
                       fit_composite, paired_auroc, rank_scores, score_eligibility, score_target_matrix)
from .charts import (PNG_CONFIG, curves, distribution, kinetics, layout, method_agreement, ranking_chart,
                     score_distribution, stratified_distribution)
from .experiments import attach_experiments, sequence_labels
from .registry import CONSENSUS_MODES, builtin_composites

OUTCOME_LABELS = {"binds": "Binding / binary", "binding_ord": "Binding / ordinal", "log10_kd": "Kd / internal"}


@st.cache_data(show_spinner=False, max_entries=16)
def cached_ranking(frame, catalog, outcome, metric, include_unconfirmed, n_boot, seed, top_fraction):
    return rank_scores(frame, catalog, outcome, metric, include_unconfirmed, n_boot=n_boot, seed=seed, top_fraction=top_fraction)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_score_targets(frame, columns, directions, outcome, metric, minimum_sequences, n_boot, seed, top_fraction):
    return score_target_matrix(frame, list(columns), dict(directions), outcome, metric, minimum_sequences,
                               n_boot=n_boot, seed=seed, top_fraction=top_fraction)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_combinations(frame, columns, directions, outcome, metric, max_scores, within_target, n_boot, seed, top_fraction):
    return combination_gain(frame, list(columns), dict(directions), outcome, metric, max_scores, within_target,
                            n_boot=n_boot, seed=seed, top_fraction=top_fraction)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_fit(frame, columns, outcome, seed):
    return fit_composite(frame, columns, outcome, seed)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_strata(frame, scores, splits, directions, outcome, metric, n_boot, seed, top_fraction):
    directions = dict(directions)
    rows = []
    for column in [value for value in splits if value in {"target", "source"}]:
        for value, subset in frame.groupby(column):
            for score in scores:
                rows.append({"stratum": f"{column}={value}",
                             **evaluate(subset, score, outcome, metric, directions[score], n_boot, seed, top_fraction)})
    if "model" in splits:
        for score in scores:
            rows.append({"stratum": f"model={score.rsplit('__', 1)[-1]}",
                         **evaluate(frame, score, outcome, metric, directions[score], n_boot, seed, top_fraction)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_scopes(frame, sequence_scores, scores, directions, outcome, metric, n_boot, seed, top_fraction):
    directions = dict(directions)
    rows = []
    for score in scores:
        within = frame.copy()
        rank_values = sequence_scores[score].groupby(sequence_scores.target).rank(pct=True)
        lookup = pd.Series(rank_values.to_numpy(), index=sequence_scores.sequence_id)
        within[score] = within.sequence_id.map(lookup)
        for label, subset in [("Across-target values", frame), ("Within-target ranks", within)]:
            rows.append({"scope": label,
                         **evaluate(subset, score, outcome, metric, directions[score], n_boot, seed, top_fraction)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_paired_auroc(frame, score_a, score_b, directions, n_boot, seed):
    return pd.DataFrame([paired_auroc(frame, score_a, score_b, dict(directions), n_boot, seed)])


def remembered(key, context, compute, run, spinner="Computing"):
    """Store a computed result with the inputs it was derived from, and forget it once those inputs change."""
    if run:
        with st.spinner(spinner):
            st.session_state[key] = (context, compute())
    stored = st.session_state.get(key)
    return stored[1] if stored is not None and stored[0] == context else None


def fingerprint(frame):
    """Cheap stand-in for the row identity of a frame, so result contexts stay small."""
    return len(frame), int(pd.util.hash_pandas_object(frame.sequence_id, index=False).sum())


def ranking_context(table, evidence, settings, config):
    return (settings.outcome, settings.metric, st.session_state.get("dataset_key"), fingerprint(table),
            tuple(settings.scoped_catalog.column), settings.include_inferred, settings.pooled_sources,
            config.bootstrap_samples, config.random_seed, settings.top_fraction,
            fingerprint(evidence), config.to_yaml())


def derived_scores(table, catalog, config, experiments):
    table = table.copy()
    catalog = catalog.copy()
    definitions = {entry["column"]: entry for entry in catalog.to_dict("records")}
    directions = {column: entry["direction"] for column, entry in definitions.items() if entry["direction"] != "unknown"}
    models = sorted({folder.model for folder in config.folders})
    specs = builtin_composites(models, directions) + config.composites
    coefficients = {}
    issues = []
    for spec in specs:
        column = spec["column"]
        try:
            ingredients = list(spec.get("weights", {})) if spec["kind"] == "composite" else spec.get("columns", [])
            if spec["kind"] == "composite":
                table[column] = composite(table, spec["weights"], directions, spec.get("normalization", "rank"), spec.get("within_target", True))
            elif spec["kind"] == "consensus":
                table[column] = consensus(table, spec["columns"], directions, spec.get("mode", "mean_rank"),
                                          spec.get("threshold", 0.8), spec.get("minimum_models", 1), spec.get("within_target", True))
            elif spec["kind"] == "fitted":
                if experiments is None or experiments.empty:
                    raise ValueError("Fitted composite has no experimental labels")
                evidence = attach_experiments(table, experiments)
                scores, fitted = cached_fit(evidence, spec["columns"], spec.get("outcome", "binds"), config.random_seed)
                lookup = pd.Series(scores.to_numpy(), index=evidence.sequence_id).groupby(level=0).first()
                table[column] = table.sequence_id.map(lookup)
                coefficients[column] = fitted
            else:
                raise ValueError("Unsupported composite kind")
            confirmed = spec["kind"] == "fitted" or all(definitions.get(item, {}).get("confirmed", False) for item in ingredients)
            entry = {"column": column, "name": column.rsplit("__", 1)[0], "stage": "derived", "model": column.rsplit("__", 1)[-1],
                     "direction": "higher_better", "group": spec["kind"], "inferred": not confirmed, "confirmed": confirmed,
                     "missing_fraction": float(table[column].isna().mean()), "constant": table[column].nunique(dropna=True) <= 1}
            definitions[column], directions[column] = entry, "higher_better"
        except (ValueError, KeyError) as error:
            issues.append(f"{column}: {error}")
    catalog = pd.DataFrame(list(definitions.values()), columns=catalog.columns)
    return table, catalog, coefficients, issues


def composite_controls(config, catalog):
    with st.expander("Composites"):
        kinds = ["Weighted", "Consensus", "Fitted"]
        kind = st.radio("Composite type", kinds, horizontal=True, key="analysis_composite_kind")
        available = catalog.column.tolist()
        if kind != "Fitted":
            available = catalog[catalog.direction.ne("unknown")].column.tolist()
        selected = st.multiselect("Component scores", available, key=f"analysis_components_{kind}")
        name = st.text_input("Composite name", "custom", key="analysis_composite_name")
        spec = {"column": f"composite_{re.sub(r'[^a-zA-Z0-9_]+', '_', name)}__custom", "columns": selected}
        if kind == "Weighted":
            weights = st.data_editor(pd.DataFrame({"score": selected, "weight": [1.0] * len(selected)}), hide_index=True,
                                     disabled=["score"], key=f"analysis_weights_{'|'.join(selected)}")
            spec.update(kind="composite", weights=dict(zip(weights.score, weights.weight)),
                        normalization=st.selectbox("Normalization", ["rank", "zscore"]),
                        within_target=st.checkbox("Normalize within target", True))
        elif kind == "Consensus":
            spec.update(kind="consensus", mode=st.selectbox("Consensus rule", CONSENSUS_MODES),
                        threshold=st.slider("Passing percentile", 0.0, 1.0, 0.8, 0.05),
                        minimum_models=int(st.number_input("Minimum models present", 1, max(1, len(selected)), max(1, len(selected)))))
        else:
            spec.update(kind="fitted", outcome=st.selectbox("Fitted outcome", list(OUTCOMES)))
        if st.button("Add composite", icon=":material/add:", disabled=not selected or not name.strip()):
            config.composites = [item for item in config.composites if item["column"] != spec["column"]] + [spec]
            st.rerun()
        if config.composites:
            remove = st.multiselect("Remove composites", [item["column"] for item in config.composites])
            if st.button("Remove", icon=":material/delete:", disabled=not remove):
                config.composites = [item for item in config.composites if item["column"] not in remove]
                st.rerun()


def exploration_observations(table, scores):
    observations = table.drop_duplicates("sequence_id").copy()
    means = table.groupby("sequence_id", sort=False)[scores].mean()
    observations[scores] = means.reindex(observations.sequence_id).to_numpy()
    return observations


def render_ranking_quality(evidence, available, catalog, config):
    outcomes = [outcome for outcome in OUTCOMES if outcome in evidence and evidence[outcome].notna().any()]
    if not outcomes:
        st.info("Load experimental measurements to compare how well scores rank designs.")
        return
    controls = st.columns(3)
    outcome = controls[0].selectbox("Outcome", outcomes, format_func=OUTCOME_LABELS.get, key="analysis_quality_outcome")
    metric = controls[1].selectbox("Ranking metric", OUTCOMES[outcome], key="analysis_quality_metric")
    minimum = int(controls[2].number_input("Minimum sequences per target", 2, 500, 5, key="analysis_quality_minimum"))
    columns = st.multiselect("Scores to compare", available, default=available[:8], key="analysis_quality_scores")
    top_fraction = st.slider("Top fraction", 0.01, 0.5, 0.1, 0.01, key="analysis_quality_fraction") if metric in {"Hit rate", "Enrichment"} else 0.1
    frame = evidence[evidence.source.eq("internal")] if outcome == "log10_kd" and "source" in evidence else evidence
    directions = tuple(sorted(zip(catalog.column, catalog.direction)))
    context = (tuple(columns), outcome, metric, minimum, top_fraction, directions, config.bootstrap_samples,
               config.random_seed, fingerprint(frame))
    run = bool(columns) and st.button("Compare ranking quality", icon=":material/leaderboard:", key="analysis_quality_run")
    matrix = remembered("analysis_quality", context,
                        lambda: cached_score_targets(frame, tuple(columns), directions, outcome, metric, minimum,
                                                     config.bootstrap_samples, config.random_seed, top_fraction),
                        run, "Scoring targets")
    if matrix is None:
        return
    if matrix.empty or matrix.value.isna().all():
        st.info("No score and target combination has enough labelled sequences.")
        return
    overall = matrix[matrix.scope.eq("Overall")].sort_values("value", ascending=False, na_position="last")
    st.caption(f"Global {metric} per score; targets with fewer than {minimum} labelled sequences are left blank.")
    st.dataframe(overall[["score", "value", "ci_low", "ci_high", "n", "positives", "negatives"]], hide_index=True)
    pivot = matrix.pivot_table(index="score", columns="scope", values="value")
    pivot = pivot.reindex([column for column in overall.score if column in pivot.index])
    figure = px.imshow(pivot, color_continuous_scale="Viridis", aspect="auto", text_auto=".2f")
    st.plotly_chart(layout(figure, f"{metric} by score and target", max(320, len(pivot) * 32 + 150)),
                    config=PNG_CONFIG, key="analysis_quality_heatmap")
    with st.expander("Per-target detail"):
        st.dataframe(matrix[["scope", "score", "value", "ci_low", "ci_high", "n", "positives", "negatives"]],
                     hide_index=True)


def render_exploration(table, full_table, catalog, config, experiments):
    available = [column for column in catalog.column if column in table and pd.api.types.is_numeric_dtype(table[column]) and table[column].notna().any()]
    if not available:
        return
    st.subheader("Explore Predictions")
    preferred = next((column for column in available if column.startswith("composite__")), available[0])
    score = st.selectbox("Exploration score", available, index=available.index(preferred), key="analysis_explore_score")
    direction = catalog.set_index("column").loc[score, "direction"]
    full = exploration_observations(full_table, available)
    active = exploration_observations(table, available)
    evidence = attach_experiments(active, experiments.table) if experiments is not None and not experiments.table.empty else active
    identifiers = [column for column in ["experimental_name", "experimental_id"] if column in evidence]
    if identifiers:
        identities = evidence.groupby("sequence_id", sort=False)[identifiers].first()
        for observations in [full, active]:
            for column in identifiers:
                observations[column] = observations.sequence_id.map(identities[column])
    names = pd.Series(sequence_labels(evidence).to_numpy(), index=evidence.sequence_id).groupby(level=0).first()
    selected = st.session_state.get("selected_candidates", pd.DataFrame())
    selected_ids = selected.sequence_id.tolist() if "sequence_id" in selected else []
    selected_ids = [value for value in selected_ids if value in set(active.sequence_id)]
    defaults = selected_ids[:4] or active.sort_values(score, ascending=direction == "lower_better").sequence_id.head(4).tolist()
    highlights = st.multiselect("Highlighted sequences", active.sequence_id.tolist(), default=defaults,
                               max_selections=8, key="analysis_explore_highlights", format_func=lambda value: names.get(value, value))
    distributions, agreement, ranking = st.tabs(["Distributions", "Model Agreement", "Ranking Quality"])
    with distributions:
        replicate_rows = st.toggle("Retained replicate rows", key="analysis_explore_replicates")
        reference, current = (full_table, table) if replicate_rows else (full, active)
        st.plotly_chart(score_distribution(reference, current, score, () if replicate_rows else highlights, replicate_rows),
                        config=PNG_CONFIG, key="analysis_explore_distribution")
        groups = [column for column in ["target", "source", "binding_class"] if column in evidence and evidence[column].notna().any()]
        groups += [column for column in evidence if column.lower().startswith("fsec") and evidence[column].notna().any()]
        if groups:
            group = st.selectbox("Distribution group", groups, key="analysis_explore_group")
            st.plotly_chart(stratified_distribution(evidence, score, group),
                            config=PNG_CONFIG, key="analysis_explore_strata")
    with agreement:
        axes = st.columns(2)
        horizontal = axes[0].selectbox("X score", available, index=available.index(preferred), key="analysis_agreement_x")
        vertical_default = next((column for column in available if column != horizontal and column.rsplit("__", 1)[0] == horizontal.rsplit("__", 1)[0]),
                                next((column for column in available if column != horizontal), horizontal))
        vertical = axes[1].selectbox("Y score", available, index=available.index(vertical_default), key="analysis_agreement_y")
        color_by = st.radio("Color by", ["binding_class", "binds"], horizontal=True, key="analysis_agreement_color",
                            format_func=lambda value: "Binding strength" if value == "binding_class" else "Binding")
        st.plotly_chart(method_agreement(evidence, horizontal, vertical, highlights, color_by),
                        config=PNG_CONFIG, key="analysis_explore_agreement")
    with ranking:
        render_ranking_quality(evidence, available, catalog, config)


class Settings(NamedTuple):
    outcome: str
    metric: str
    top_fraction: float
    include_inferred: bool
    pooled_sources: bool
    scoped_catalog: pd.DataFrame
    directions: dict


def analysis_controls(catalog, experiments, evidence, config):
    controls = st.columns([1, 1, 1])
    outcome = controls[0].selectbox("Outcome", list(OUTCOMES), format_func=OUTCOME_LABELS.get, key="analysis_outcome")
    metric = controls[1].selectbox("Metric", OUTCOMES[outcome], key="analysis_metric")
    config.bootstrap_samples = int(controls[2].number_input("Bootstrap samples", 50, 5000, int(config.bootstrap_samples), 50, key="analysis_bootstrap"))
    config.random_seed = int(st.number_input("Random seed", 0, 2147483647, config.random_seed, key="analysis_seed"))
    include_inferred = st.checkbox("Include unconfirmed inferred directions", True, key="analysis_include_inferred",
                                   help="Directions inferred from the data are used unless you restrict the analysis to registry-confirmed scores.")
    with st.expander("Score Filters"):
        choices = st.columns(3)
        stages = choices[0].multiselect("Stages", sorted(catalog.stage.unique()), key="analysis_filter_stages")
        groups = choices[1].multiselect("Groups", sorted(catalog.group.fillna("other").unique()), key="analysis_filter_groups")
        models = choices[2].multiselect("Models", sorted(catalog.model.unique()), key="analysis_filter_models")
        scoped_catalog = catalog.copy()
        for column, values in [("stage", stages), ("group", groups), ("model", models)]:
            if values:
                scoped_catalog = scoped_catalog[scoped_catalog[column].isin(values)]
        eligibility = score_eligibility(evidence, scoped_catalog, include_inferred)
        st.caption(f"{int(eligibility.eligible.sum())} of {len(eligibility)} scores are eligible with the current filters.")
        st.dataframe(eligibility, hide_index=True)
    top_fraction = st.slider("Top fraction", 0.01, 0.5, 0.1, 0.01, key="analysis_top_fraction") if metric in {"Hit rate", "Enrichment"} else 0.1
    pooled_sources = st.toggle("Pooled class metrics", value=True, disabled=outcome == "log10_kd", key="analysis_pooled")
    if outcome == "log10_kd":
        evidence = evidence[evidence.source.eq("internal")]
        st.info("Censored Kd values contribute their recorded bounds with tied ranks; this is a bound-based Spearman analysis, not a survival model.")
    elif not pooled_sources:
        source = st.selectbox("Experimental source", sorted(experiments.table.source.unique()), key="analysis_source")
        evidence = evidence[evidence.source.eq(source)]
    return evidence, Settings(outcome, metric, top_fraction, include_inferred, pooled_sources, scoped_catalog,
                              dict(zip(catalog.column, catalog.direction)))


def render_correlations(table, ranking):
    with st.expander("Score Correlations"):
        columns = ranking.score.head(30).tolist()
        correlations = table.groupby("sequence_id", sort=False)[columns].mean().corr(method="spearman")
        figure = px.imshow(correlations, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(layout(figure, "Spearman correlations", 550), config=PNG_CONFIG)


def render_combinations(evidence, ranking, settings, config):
    with st.expander("Score Combinations"):
        st.caption("Greedy forward search: the best single score, then the score that adds the most when combined as "
                   "equally weighted percentile ranks. Lift is the gain over the best single score.")
        pool = st.multiselect("Candidate scores", ranking.score.tolist(), default=ranking.score.head(6).tolist(), key="analysis_combination_pool")
        controls = st.columns(2)
        max_scores = int(controls[0].number_input("Maximum scores combined", 2, 6, 3, key="analysis_combination_max"))
        within_target = controls[1].checkbox("Rank within each target", True, key="analysis_combination_within")
        context = (tuple(pool), settings.outcome, settings.metric, max_scores, within_target, settings.top_fraction,
                   config.bootstrap_samples, config.random_seed, fingerprint(evidence))
        run = len(pool) >= 2 and st.button("Search combinations", icon=":material/travel_explore:", key="analysis_combination_run")
        gains = remembered("analysis_combinations", context,
                           lambda: cached_combinations(evidence, tuple(pool), tuple(sorted(settings.directions.items())),
                                                       settings.outcome, settings.metric, max_scores, within_target,
                                                       config.bootstrap_samples, config.random_seed, settings.top_fraction),
                           run, "Searching score combinations")
        if gains is None or gains.empty:
            return
        st.dataframe(gains, hide_index=True)
        figure = px.line(gains, x="n_scores", y="value", markers=True, hover_data=["scores"])
        figure.update_xaxes(dtick=1)
        st.plotly_chart(layout(figure, f"{settings.metric} versus number of combined scores", 380),
                        config=PNG_CONFIG, key="analysis_combination_chart")
        best = gains.loc[gains.value.idxmax()]
        st.metric(f"Best {settings.metric}", f"{best.value:.3f}", delta=f"{best.lift:+.3f} vs best single score")
        st.caption(f"Best combination: {best.scores}")


def render_stratification(table, evidence, scores, settings, config):
    with st.expander("Stratification"):
        splits = st.multiselect("Split by", ["target", "source", "model"], default=["target", "source"], key="analysis_strata_splits")
        directions = tuple(sorted((score, settings.directions[score]) for score in scores))
        context = (settings.outcome, settings.metric, settings.top_fraction, config.bootstrap_samples,
                   config.random_seed, directions, fingerprint(evidence))
        run_strata = st.button("Compute strata", icon=":material/account_tree:", key="analysis_strata_run")
        strata = remembered("analysis_strata", (tuple(splits),) + context,
                            lambda: cached_strata(evidence, tuple(scores), tuple(splits), directions, settings.outcome,
                                                  settings.metric, config.bootstrap_samples, config.random_seed, settings.top_fraction),
                            run_strata, "Evaluating strata")
        if strata is not None:
            st.dataframe(strata, hide_index=True)
        run_scopes = st.button("Within-target vs across-target", icon=":material/compare_arrows:", key="analysis_scopes_run")
        sequence_scores = exploration_observations(table, list(scores))
        comparison = remembered("analysis_scopes", context,
                                lambda: cached_scopes(evidence, sequence_scores, tuple(scores), directions, settings.outcome,
                                                      settings.metric, config.bootstrap_samples, config.random_seed, settings.top_fraction),
                                run_scopes, "Comparing ranking scopes")
        if comparison is not None:
            st.dataframe(comparison, hide_index=True)


def render_pairwise(evidence, scores, settings, config):
    with st.expander("A vs B"):
        left, right = st.columns(2)
        score_a = left.selectbox("Score A", scores, key="analysis_pair_a")
        score_b = right.selectbox("Score B", scores, index=1, key="analysis_pair_b")
        paired = evidence.dropna(subset=[score_a, score_b]).drop_duplicates("sequence_id")
        st.plotly_chart(method_agreement(paired, score_a, score_b), config=PNG_CONFIG, key="analysis_pair_agreement")
        if settings.outcome != "binds":
            return
        st.plotly_chart(curves(paired, [score_a, score_b], settings.directions), config=PNG_CONFIG, key="analysis_pair_roc")
        directions = tuple(sorted((score, settings.directions[score]) for score in [score_a, score_b]))
        context = (score_a, score_b, directions, config.bootstrap_samples, config.random_seed, fingerprint(paired))
        run = st.button("Paired AUROC comparison", icon=":material/compare:", key="analysis_pair_run")
        difference = remembered("analysis_pair", context,
                                lambda: cached_paired_auroc(paired, score_a, score_b, directions,
                                                            config.bootstrap_samples, config.random_seed),
                                run, "Bootstrapping the AUROC difference")
        if difference is not None:
            st.dataframe(difference, hide_index=True)


def render_sensitivity(inspected, experiments, settings, config, result, active_config):
    with st.expander("Aggregation Sensitivity"):
        context = (inspected, settings.outcome, settings.metric, settings.top_fraction, config.bootstrap_samples,
                   config.random_seed, st.session_state.get("dataset_key"))
        run = st.button("Compare best / mean / worst", icon=":material/science:", key="analysis_sensitivity_run",
                        disabled=inspected not in result.table)
        sensitivity = remembered("analysis_sensitivity", context,
                                 lambda: aggregation_sensitivity(active_config, inspected, settings.outcome,
                                                                 settings.metric, settings.directions[inspected], experiments.table,
                                                                 n_boot=config.bootstrap_samples, seed=config.random_seed,
                                                                 top_fraction=settings.top_fraction),
                                 run, "Re-aggregating replicates")
        if sensitivity is not None:
            st.dataframe(sensitivity, hide_index=True)


def render_analysis(table, catalog, config, experiments, coefficients, result, active_config, full_table=None):
    composite_controls(config, catalog)
    render_exploration(table, full_table if full_table is not None else table, catalog, config, experiments)
    if experiments is None or experiments.table.empty:
        st.info("No experimental measurements loaded.")
        return
    evidence, settings = analysis_controls(catalog, experiments, attach_experiments(table, experiments.table), config)
    context = ranking_context(table, evidence, settings, config)
    run = st.button("Analyze", type="primary", icon=":material/analytics:")
    ranking = remembered("analysis_ranking", context,
                         lambda: cached_ranking(evidence, settings.scoped_catalog, settings.outcome, settings.metric,
                                                settings.include_inferred, config.bootstrap_samples, config.random_seed,
                                                settings.top_fraction),
                         run, "Bootstrapping sequences")
    if ranking is None:
        return
    if run and not ranking.empty and settings.outcome == "binds" and settings.metric == "AUROC" and pd.notna(ranking.iloc[0].value):
        st.session_state["best_score"] = ranking.iloc[0].score
    if ranking.empty:
        reasons = score_eligibility(evidence, settings.scoped_catalog, settings.include_inferred)
        st.info("No score is eligible for this analysis. Expand Score Filters to see why each score was excluded.")
        st.dataframe(reasons[~reasons.eligible], hide_index=True)
        return
    if ranking.small_n.any():
        st.warning("Small sequence or class counts: confidence intervals may be unstable.")
    st.dataframe(ranking[["score", "value", "ci_low", "ci_high", "n", "positives", "negatives", "valid_bootstraps", "inferred", "confirmed"]], hide_index=True)
    st.plotly_chart(ranking_chart(ranking), config=PNG_CONFIG)
    scores = st.multiselect("Inspect scores", ranking.score.tolist(), default=ranking.score.head(2).tolist(), key="analysis_inspect")
    if not scores:
        return
    inspected = scores[0]
    if settings.outcome == "binds":
        panels = st.columns(2)
        for panel, kind in zip(panels, ["ROC", "PR"]):
            panel.plotly_chart(curves(evidence, scores, settings.directions, kind), config=PNG_CONFIG)
    if settings.outcome == "log10_kd":
        st.plotly_chart(kinetics(evidence, inspected), config=PNG_CONFIG)
    else:
        st.plotly_chart(distribution(evidence, inspected), config=PNG_CONFIG)
    render_correlations(table, ranking)
    render_combinations(evidence, ranking, settings, config)
    render_stratification(table, evidence, scores, settings, config)
    if len(scores) >= 2:
        render_pairwise(evidence, scores, settings, config)
    with st.expander("Fitted Coefficients"):
        for name, fitted in coefficients.items():
            st.markdown(f"**{name}**")
            st.dataframe(fitted, hide_index=True)
    render_sensitivity(inspected, experiments, settings, config, result, active_config)