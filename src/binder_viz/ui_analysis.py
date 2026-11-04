from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .analysis import OUTCOMES, aggregation_sensitivity, composite, consensus, evaluate, fit_composite, paired_auroc, rank_scores
from .charts import (PNG_CONFIG, candidate_nomination_budget, cluster_distribution, curves, distribution,
                     kd_scatter, kinetics, layout, method_agreement, ranking_chart, score_distribution,
                     selected_method_blocks, stratified_distribution)
from .experiments import attach_experiments, sequence_labels


@st.cache_data(show_spinner=False, max_entries=16)
def cached_ranking(frame, catalog, outcome, metric, include_unconfirmed, n_boot, seed, top_fraction):
    return rank_scores(frame, catalog, outcome, metric, include_unconfirmed, n_boot=n_boot, seed=seed, top_fraction=top_fraction)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_fit(frame, columns, outcome, seed):
    return fit_composite(frame, columns, outcome, seed)


def derived_scores(table, catalog, config, experiments):
    table = table.copy()
    catalog = catalog.copy()
    definitions = {entry["column"]: entry for entry in catalog.to_dict("records")}
    directions = {column: entry["direction"] for column, entry in definitions.items() if entry["direction"] != "unknown"}
    specs = []
    models = sorted({folder.model for folder in config.folders})
    for model in models:
        confidence = [column for column in [f"iptm__{model}", f"ipsae__{model}"] if column in directions]
        if len(confidence) == 2:
            specs.append({"column": f"composite_confidence__{model}", "kind": "composite", "weights": dict.fromkeys(confidence, 1.0), "normalization": "rank"})
        interface = [f"iptm__{model}", f"ros_interface_dG__{model}"]
        if all(column in directions for column in interface):
            specs.append({"column": f"composite_interface__{model}", "kind": "composite", "weights": dict.fromkeys(interface, 1.0), "normalization": "rank"})
    iptms = [f"iptm__{model}" for model in models if f"iptm__{model}" in directions]
    if len(iptms) > 1:
        for mode in ["mean_rank", "mean_zscore", "min", "passing"]:
            specs.append({"column": f"consensus_{mode}__all", "kind": "consensus", "columns": iptms, "mode": mode, "minimum_models": len(iptms)})
    specs += config.composites
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
            weights = st.data_editor(pd.DataFrame({"score": selected, "weight": [1.0] * len(selected)}), hide_index=True, disabled=["score"], key="analysis_weights")
            spec.update(kind="composite", weights=dict(zip(weights.score, weights.weight)),
                        normalization=st.selectbox("Normalization", ["rank", "zscore"]),
                        within_target=st.checkbox("Normalize within target", True))
        elif kind == "Consensus":
            spec.update(kind="consensus", mode=st.selectbox("Consensus rule", ["mean_rank", "mean_zscore", "min", "passing"]),
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
    distributions, agreement, blocks, affinity = st.tabs(["Distributions", "Model Agreement", "Evidence Blocks", "Affinity / Ranking"])
    with distributions:
        replicate_rows = st.toggle("Retained replicate rows", key="analysis_explore_replicates")
        reference, current = (full_table, table) if replicate_rows else (full, active)
        st.plotly_chart(score_distribution(reference, current, score, () if replicate_rows else highlights, replicate_rows),
                        use_container_width=True, config=PNG_CONFIG, key="analysis_explore_distribution")
        groups = [column for column in ["target", "source", "binding_class"] if column in evidence and evidence[column].notna().any()]
        groups += [column for column in evidence if column.lower().startswith("fsec") and evidence[column].notna().any()]
        if groups:
            group = st.selectbox("Distribution group", groups, key="analysis_explore_group")
            st.plotly_chart(stratified_distribution(evidence, score, group), use_container_width=True,
                            config=PNG_CONFIG, key="analysis_explore_strata")
    with agreement:
        axes = st.columns(2)
        horizontal = axes[0].selectbox("X score", available, index=available.index(preferred), key="analysis_agreement_x")
        vertical_default = next((column for column in available if column != horizontal and column.rsplit("__", 1)[0] == horizontal.rsplit("__", 1)[0]),
                                next((column for column in available if column != horizontal), horizontal))
        vertical = axes[1].selectbox("Y score", available, index=available.index(vertical_default), key="analysis_agreement_y")
        color_by = st.radio("Color by", ["binding_class", "binds"], horizontal=True, key="analysis_agreement_color",
                            format_func=lambda value: "Binding strength" if value == "binding_class" else "Binding")
        st.plotly_chart(method_agreement(evidence, horizontal, vertical, highlights, color_by), use_container_width=True,
                        config=PNG_CONFIG, key="analysis_explore_agreement")
    with blocks:
        models = sorted({folder.model for folder in config.folders})
        st.plotly_chart(selected_method_blocks(active, highlights, models), use_container_width=True,
                        config=PNG_CONFIG, key="analysis_explore_blocks")
        if not selected.empty and "cluster_id" in selected:
            current_selection = selected[selected.sequence_id.isin(active.sequence_id)]
            st.plotly_chart(cluster_distribution(current_selection), use_container_width=True,
                            config=PNG_CONFIG, key="analysis_explore_diversity")
    with affinity:
        st.plotly_chart(kd_scatter(evidence, score), use_container_width=True, config=PNG_CONFIG, key="analysis_explore_kd")
        scope = st.radio("Ranking scope", ["Current filtered set", "Selected candidates"], horizontal=True, key="analysis_explore_scope")
        ranked = active if scope == "Current filtered set" else active[active.sequence_id.isin(selected_ids)]
        count = st.number_input("Ranking size", 1, 50, 20, key="analysis_explore_count")
        if direction == "unknown":
            direction = st.selectbox("Ranking direction", ["higher_better", "lower_better"], key="analysis_explore_direction")
        st.plotly_chart(candidate_nomination_budget(ranked, score, direction, highlights, int(count)), use_container_width=True,
                        config=PNG_CONFIG, key="analysis_explore_candidates")


def render_analysis(table, catalog, config, experiments, coefficients, full_table=None):
    composite_controls(config, catalog)
    render_exploration(table, full_table if full_table is not None else table, catalog, config, experiments)
    if experiments is None or experiments.table.empty:
        st.info("No experimental measurements loaded.")
        return
    evidence = attach_experiments(table, experiments.table)
    controls = st.columns([1, 1, 1])
    outcome = controls[0].selectbox("Outcome", list(OUTCOMES), format_func=lambda value: {"binds": "Binding / binary", "binding_ord": "Binding / ordinal", "log10_kd": "Kd / internal"}[value])
    metric = controls[1].selectbox("Metric", OUTCOMES[outcome])
    config.bootstrap_samples = int(controls[2].number_input("Bootstrap samples", 50, 5000, int(config.bootstrap_samples), 50))
    config.random_seed = int(st.number_input("Random seed", 0, 2147483647, config.random_seed))
    include_inferred = st.checkbox("Include unconfirmed inferred directions", False)
    with st.expander("Score Filters"):
        choices = st.columns(3)
        stages = choices[0].multiselect("Stages", sorted(catalog.stage.unique()))
        groups = choices[1].multiselect("Groups", sorted(catalog.group.fillna("other").unique()))
        models = choices[2].multiselect("Models", sorted(catalog.model.unique()))
        scoped_catalog = catalog.copy()
        for column, values in [("stage", stages), ("group", groups), ("model", models)]:
            if values:
                scoped_catalog = scoped_catalog[scoped_catalog[column].isin(values)]
    top_fraction = st.slider("Top fraction", 0.01, 0.5, 0.1, 0.01) if metric in {"Hit rate", "Enrichment"} else 0.1
    pooled_sources = st.toggle("Pooled class metrics", value=True, disabled=outcome == "log10_kd")
    if outcome == "log10_kd":
        evidence = evidence[evidence.source.eq("internal")]
        st.info("Censored Kd values contribute their recorded bounds with tied ranks; this is a bound-based Spearman analysis, not a survival model.")
    elif not pooled_sources:
        source = st.selectbox("Experimental source", sorted(experiments.table.source.unique()))
        evidence = evidence[evidence.source.eq(source)]
    directions = dict(zip(catalog.column, catalog.direction))
    if st.button("Analyze", type="primary", icon=":material/analytics:"):
        with st.spinner("Bootstrapping sequences"):
            ranking = cached_ranking(evidence, scoped_catalog, outcome, metric, include_inferred, config.bootstrap_samples, config.random_seed, top_fraction)
        st.session_state["analysis_ranking"] = ranking
        st.session_state["analysis_context"] = (outcome, metric, st.session_state.get("dataset_key"), tuple(table.sequence_id), tuple(scoped_catalog.column), include_inferred, pooled_sources,
                                                 config.bootstrap_samples, config.random_seed, top_fraction, tuple(evidence.get("source", [])), tuple(config.to_yaml()))
        if not ranking.empty and outcome == "binds" and metric == "AUROC" and pd.notna(ranking.iloc[0].value):
            st.session_state["best_score"] = ranking.iloc[0].score
    ranking = st.session_state.get("analysis_ranking")
    current_context = (outcome, metric, st.session_state.get("dataset_key"), tuple(table.sequence_id), tuple(scoped_catalog.column), include_inferred, pooled_sources,
                       config.bootstrap_samples, config.random_seed, top_fraction, tuple(evidence.get("source", [])), tuple(config.to_yaml()))
    if ranking is None or st.session_state.get("analysis_context") != current_context:
        return
    if ranking.empty:
        st.info("No confirmed, nonconstant scores are eligible.")
        return
    if ranking.small_n.any():
        st.warning("Small sequence or class counts: confidence intervals may be unstable.")
    st.dataframe(ranking[["score", "value", "ci_low", "ci_high", "n", "positives", "negatives", "valid_bootstraps", "inferred", "confirmed"]], hide_index=True, use_container_width=True)
    st.plotly_chart(ranking_chart(ranking), use_container_width=True, config=PNG_CONFIG)
    scores = st.multiselect("Inspect scores", ranking.score.tolist(), default=ranking.score.head(2).tolist())
    if not scores:
        return
    inspected = scores[0]
    if outcome == "binds":
        panels = st.columns(2)
        for panel, kind in zip(panels, ["ROC", "PR"]):
            panel.plotly_chart(curves(evidence, scores, directions, kind), use_container_width=True, config=PNG_CONFIG)
    if outcome == "log10_kd":
        st.plotly_chart(kinetics(evidence, inspected), use_container_width=True, config=PNG_CONFIG)
    else:
        st.plotly_chart(distribution(evidence, inspected), use_container_width=True, config=PNG_CONFIG)
    with st.expander("Score Correlations"):
        columns = ranking.score.head(30).tolist()
        correlations = table.drop_duplicates("sequence_id")[columns].corr(method="spearman")
        figure = px.imshow(correlations, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(layout(figure, "Spearman correlations", 550), use_container_width=True, config=PNG_CONFIG)
    with st.expander("Stratification"):
        stratifications = st.multiselect("Split by", ["target", "source", "model"], default=["target", "source"])
        if st.button("Compute strata", icon=":material/account_tree:"):
            rows = []
            for column in [value for value in stratifications if value in {"target", "source"}]:
                for value, subset in evidence.groupby(column):
                    for score in scores:
                        rows.append({"stratum": f"{column}={value}", **evaluate(subset, score, outcome, metric, directions[score], config.bootstrap_samples, config.random_seed, top_fraction)})
            if "model" in stratifications:
                for score in scores:
                    rows.append({"stratum": f"model={score.rsplit('__', 1)[-1]}", **evaluate(evidence, score, outcome, metric, directions[score], config.bootstrap_samples, config.random_seed, top_fraction)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if st.button("Within-target vs across-target", icon=":material/compare_arrows:"):
            rows = []
            for score in scores:
                within = evidence.copy()
                sequence_scores = table.drop_duplicates("sequence_id")
                rank_values = sequence_scores[score].groupby(sequence_scores.target).rank(pct=True)
                lookup = pd.Series(rank_values.to_numpy(), index=sequence_scores.sequence_id)
                within[score] = within.sequence_id.map(lookup)
                for label, subset in [("Across-target values", evidence), ("Within-target ranks", within)]:
                    rows.append({"scope": label, **evaluate(subset, score, outcome, metric, directions[score], config.bootstrap_samples, config.random_seed, top_fraction)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if len(scores) >= 2:
        with st.expander("A vs B"):
            left, right = st.columns(2)
            score_a = left.selectbox("Score A", scores)
            score_b = right.selectbox("Score B", scores, index=1)
            paired = evidence.dropna(subset=[score_a, score_b]).drop_duplicates("sequence_id")
            st.plotly_chart(method_agreement(paired, score_a, score_b), use_container_width=True, config=PNG_CONFIG, key="analysis_pair_agreement")
            if outcome == "binds":
                st.plotly_chart(curves(paired, [score_a, score_b], directions), use_container_width=True, config=PNG_CONFIG, key="analysis_pair_roc")
                if st.button("Paired AUROC comparison", icon=":material/compare:"):
                    st.dataframe(pd.DataFrame([paired_auroc(paired, score_a, score_b, directions, config.bootstrap_samples, config.random_seed)]), hide_index=True)
    with st.expander("Fitted Coefficients"):
        for name, fitted in coefficients.items():
            st.markdown(f"**{name}**")
            st.dataframe(fitted, hide_index=True, use_container_width=True)
    with st.expander("Aggregation Sensitivity"):
        if st.button("Compare best / mean / worst", icon=":material/science:", disabled=inspected not in st.session_state["result"].table):
            result = aggregation_sensitivity(st.session_state["active_config"], inspected, outcome, metric, directions[inspected], experiments.table,
                                             n_boot=config.bootstrap_samples, seed=config.random_seed, top_fraction=top_fraction)
            st.dataframe(result, hide_index=True, use_container_width=True)