from __future__ import annotations

from hashlib import sha256
import json

import pandas as pd
import streamlit as st

from .charts import PNG_CONFIG, candidate_nomination_budget, cluster_distribution
from .selection import COMBINATION_RULES, COMBINED_SCORE, candidate_pool, combine_scores, export_selection, select_candidates

RULE_LABELS = {"mean_rank": "Mean rank (weighted)", "min_rank": "Minimum rank (worst score)",
               "intersection": "Intersection (all scores pass percentile)", "union": "Union (any score passes percentile)"}


@st.cache_data(show_spinner=False, max_entries=16)
def cached_selection(pool, count, identity, per_cluster, per_target, score, cluster, caps):
    return select_candidates(pool, count, identity, per_cluster, per_target, score, cluster, caps)


def ranking_controls(table, catalog, settings, scores, directions):
    """Returns the table carrying the ranking column, the ranking column, its direction, a label and the stored config."""
    preferred = [column for column in settings.get("scores", []) if column in scores]
    preferred = preferred or [column for column in [st.session_state.get("best_score"), settings.get("score", "iptm__af2")] if column in scores][:1] or scores[:1]
    selected = st.multiselect("Ranking scores", scores, default=preferred, key="selection_scores")
    if not selected:
        st.info("Select at least one ranking score.")
        return None
    confirmed = dict(zip(catalog.column, catalog.confirmed))
    unconfirmed = [column for column in selected if not confirmed.get(column, False)]
    if unconfirmed:
        st.warning(f"Unconfirmed score direction: {', '.join(unconfirmed)}.")
    if len(selected) == 1:
        score = selected[0]
        direction = st.selectbox("Ranking direction", ["higher_better", "lower_better"], index=int(directions[score] == "lower_better"))
        return table, score, direction, score, {"scores": selected, "score": score, "direction": direction}
    controls = st.columns([2, 1])
    rule = controls[0].selectbox("Combination rule", COMBINATION_RULES, format_func=RULE_LABELS.get,
                                 index=COMBINATION_RULES.index(settings.get("combination_rule", "mean_rank")) if settings.get("combination_rule") in COMBINATION_RULES else 0,
                                 key="selection_combination_rule")
    percentile = controls[1].slider("Rule percentile", 0.0, 1.0, float(settings.get("combination_percentile", 0.8)), 0.05,
                                    disabled=rule in {"mean_rank", "min_rank"}, key="selection_combination_percentile")
    stored = settings.get("combination_weights", {})
    editor = st.data_editor(pd.DataFrame({"score": selected, "weight": [float(stored.get(column, 1.0)) for column in selected]}),
                            hide_index=True, disabled=["score"], key="selection_combination_weights",
                            column_config={"weight": st.column_config.NumberColumn("Weight", min_value=0.0, step=0.1)})
    weights = {row["score"]: float(row["weight"]) for row in editor.to_dict("records")}
    within_target = st.checkbox("Rank within each target", bool(settings.get("combination_within_target", True)), key="selection_combination_within")
    table = table.copy()
    table[COMBINED_SCORE] = combine_scores(table, weights, directions, rule, percentile, within_target)
    label = f"{rule}({', '.join(selected)})"
    return table, COMBINED_SCORE, "higher_better", label, {"scores": selected, "score": COMBINED_SCORE, "direction": "higher_better",
                                                           "combination_rule": rule, "combination_percentile": percentile,
                                                           "combination_weights": weights, "combination_within_target": within_target}


def render_selection(table, catalog, config, experiments, off_target_predictions=None):
    if table.empty or catalog.empty:
        st.info("No candidates available.")
        return
    st.subheader("Candidate Selection")
    scores = catalog[catalog.direction.ne("unknown") & ~catalog.constant].column.tolist()
    if not scores:
        st.info("No nonconstant score has a favorable direction.")
        return
    settings = config.selection
    directions = dict(zip(catalog.column, catalog.direction))
    try:
        ranking = ranking_controls(table, catalog, settings, scores, directions)
    except ValueError as error:
        st.error(str(error))
        return
    if ranking is None:
        return
    table, score, direction, score_label, ranking_settings = ranking
    exclude = st.checkbox("Exclude already-tested sequences", settings.get("exclude_tested", True))
    reference = table if off_target_predictions is None else off_target_predictions
    if score == COMBINED_SCORE and COMBINED_SCORE not in reference:
        try:
            combined = combine_scores(reference, ranking_settings["combination_weights"], directions, ranking_settings["combination_rule"],
                                      ranking_settings["combination_percentile"], ranking_settings["combination_within_target"])
        except ValueError as error:
            st.error(f"Combined ranking is unavailable for off-target predictions: {error}")
            return
        reference = reference.assign(**{COMBINED_SCORE: combined})
    experiment_table = experiments.table if experiments is not None else None
    targets = sorted(table.target.unique())
    negative_options = set(reference.target.unique())
    if experiment_table is not None and not experiment_table.empty:
        negative_options.update(experiment_table.target.dropna().unique())
    with st.expander("Target Specificity / Negative Selection", expanded=True):
        positive_targets = st.multiselect("Positive targets", targets,
                                         default=[target for target in settings.get("positive_targets", []) if target in targets],
                                         key="selection_positive_targets")
        stored_rules = {rule["target"]: rule for rule in settings.get("negative_selection", [])}
        # Off-target scores and cutoffs are tied to the ranking score, so rules recorded under a different one are stale.
        previous_rules = stored_rules if settings.get("off_target_ranking_score", score_label) == score_label else {}
        negative_targets = st.multiselect("Negative targets (must not bind)", sorted(negative_options),
                                         default=[target for target in stored_rules if target in negative_options],
                                         key="selection_negative_targets")
        exclude_missing = st.checkbox("Exclude sequences without off-target evidence", settings.get("exclude_missing_off_targets", True),
                                      key="selection_exclude_missing")
        use_predictions = st.checkbox("Apply predicted off-target binding thresholds", settings.get("off_target_use_predictions", True),
                                      key="selection_negative_predictions")
        negative_rules = []
        if negative_targets and use_predictions:
            default_values = reference[score].dropna()
            default_threshold = float(default_values.median()) if not default_values.empty else 0.7
            rules = [{"target": target, "score": previous_rules.get(target, {}).get("score") or score,
                      "direction": previous_rules.get(target, {}).get("direction", direction),
                      "threshold": previous_rules.get(target, {}).get("threshold", default_threshold)} for target in negative_targets]
            score_options = list(dict.fromkeys([score, *catalog.column.tolist()]))
            editor = st.data_editor(pd.DataFrame(rules), hide_index=True, disabled=["target"],
                                    key="selection_negative_thresholds_" + sha256(json.dumps([score_label, negative_targets]).encode()).hexdigest()[:10],
                                    column_config={"target": "Negative target",
                                                   "score": st.column_config.SelectboxColumn("Off-target score", options=score_options, required=True),
                                                   "direction": st.column_config.SelectboxColumn("Binding direction", options=["higher_better", "lower_better"], required=True),
                                                   "threshold": st.column_config.NumberColumn("Binding cutoff (inclusive)", required=True)})
            negative_rules = editor.to_dict("records")
        elif negative_targets:
            negative_rules = [{"target": target} for target in negative_targets]
    with st.expander("Hard Filters"):
        thresholds = pd.DataFrame(settings.get("thresholds", []), columns=["score", "min", "max", "target"])
        thresholds = st.data_editor(thresholds, num_rows="dynamic", hide_index=True, key="selection_thresholds",
                                     column_config={"score": st.column_config.SelectboxColumn("Score", options=catalog.column.tolist()),
                                                    "min": st.column_config.NumberColumn("Minimum"), "max": st.column_config.NumberColumn("Maximum"),
                                                    "target": st.column_config.SelectboxColumn("Target", options=[""] + sorted(table.target.unique()))})
        filters = [row for row in thresholds.where(pd.notna(thresholds), None).to_dict("records") if row["score"]]
    with st.expander("Model Agreement"):
        model_scores = st.multiselect("One score per model", scores, default=[column for column in settings.get("agreement_columns", []) if column in scores])
        minimum_models = st.number_input("Models required to pass", 0, max(1, len(model_scores)), min(int(settings.get("minimum_models", 0)), len(model_scores)))
        percentile = st.slider("Minimum within-target percentile", 0.0, 1.0, float(settings.get("agreement_percentile", 0.8)), 0.05)
    columns = st.columns(3)
    count = int(columns[0].number_input("Candidates", 1, 10000, int(settings.get("count", 12))))
    per_target = int(columns[1].number_input("Maximum per target", 1, 10000, int(settings.get("per_target", 12))))
    pool_limit = int(columns[2].number_input("Maximum ranked pool", 10, 100000, int(settings.get("pool_limit", 2000)), 10))
    with st.expander("Sequence Diversity"):
        cluster = st.checkbox("Sequence identity clustering", settings.get("cluster", True))
        identity = st.slider("Cluster identity threshold", 0.1, 1.0, float(settings.get("identity", 0.8)), 0.05)
        per_cluster = int(st.number_input("Maximum per cluster", 1, 1000, int(settings.get("per_cluster", 1))))
        cap_rows = pd.DataFrame([{"target": target, "cap": settings.get("target_caps", {}).get(target, per_target)} for target in sorted(table.target.unique())])
        caps_editor = st.data_editor(cap_rows, hide_index=True, disabled=["target"], key="selection_target_caps",
                                     column_config={"cap": st.column_config.NumberColumn("Target cap", min_value=1, max_value=10000, step=1)})
        caps = {row["target"]: int(row["cap"]) for row in caps_editor.to_dict("records")}
    config.selection = {**ranking_settings, "exclude_tested": exclude, "thresholds": filters,
                        "positive_targets": positive_targets, "negative_selection": negative_rules,
                        "off_target_ranking_score": score_label,
                        "exclude_missing_off_targets": exclude_missing, "off_target_use_predictions": use_predictions,
                        "agreement_columns": model_scores, "minimum_models": int(minimum_models), "agreement_percentile": percentile,
                        "count": count, "per_target": per_target, "pool_limit": pool_limit, "cluster": cluster,
                        "identity": identity, "per_cluster": per_cluster, "target_caps": caps}
    tested = set(experiments.table.sequence_id) if exclude and experiments is not None else set()
    st.session_state["selected_candidates"] = table.iloc[:0].copy()
    try:
        pool = candidate_pool(table, score, direction, filters, tested, model_scores, directions, int(minimum_models), percentile,
                              positive_targets=positive_targets, negative_selection=negative_rules,
                              off_target_experiments=experiment_table, exclude_missing_off_targets=exclude_missing,
                              off_target_predictions=reference)
        if len(pool) > pool_limit:
            st.warning(f"Ranked pool contains {len(pool):,} sequences; clustering and manual review are limited to the top {pool_limit:,}.")
        pool = pool.head(pool_limit)
        shortlist = cached_selection(pool, count, identity, per_cluster, per_target, score_label, cluster, caps)
    except ValueError as error:
        st.error(str(error))
        return
    if pool.empty:
        st.info("No sequences pass the current filters.")
        return
    selected_ids = set(shortlist.sequence_id)
    shortlist_info = shortlist.set_index("sequence_id")
    pool["selected"] = pool.sequence_id.isin(selected_ids)
    pool["cluster_id"] = pool.sequence_id.map(shortlist_info.cluster_id).astype("Int64")
    pool["selection_rationale"] = pool.sequence_id.map(shortlist_info.selection_rationale).fillna("Manual addition")
    display = list(dict.fromkeys(["selected", "sequence_id", "target", "selection_rank", score, *ranking_settings["scores"],
                                  "cluster_id", "n_rep_p2", "n_rep_p3", "flags", "selection_rationale"]
                                + [column for column in pool if column.startswith(("rank__", "off_target__"))]))
    display = [column for column in display if column in pool]
    identity_key = sha256((json.dumps(config.selection, sort_keys=True) + st.session_state.get("dataset_key", "") + str(list(table.sequence_id))).encode()).hexdigest()[:14]
    manual = st.data_editor(pool[display], hide_index=True, key=f"selection_editor_{identity_key}",
                            disabled=[column for column in display if column not in {"selected", "selection_rationale"}],
                            column_config={"selected": st.column_config.CheckboxColumn("Pick"), "selection_rationale": st.column_config.TextColumn("Rationale", width="large")})
    chosen_ids = set(manual.loc[manual.selected, "sequence_id"])
    selected = pool[pool.sequence_id.isin(chosen_ids)].copy()
    selected["selected"] = True
    selected["selection_rationale"] = selected.sequence_id.map(manual.set_index("sequence_id").selection_rationale)
    selected["manual_override"] = ~selected.sequence_id.isin(selected_ids)
    if selected.manual_override.any():
        st.warning("Manual additions bypass automatic cluster and target caps.")
    st.metric("Selected sequences", len(selected))
    config.selection["manual_selected_ids"] = sorted(chosen_ids)
    config.selection["manual_rationale"] = dict(zip(selected.sequence_id, selected.selection_rationale))
    st.session_state["selected_candidates"] = selected
    shown = int(st.number_input("Ranked candidates charted", 1, 100, min(20, len(pool)), 5, key="selection_chart_count"))
    charts = st.columns(2)
    charts[0].plotly_chart(candidate_nomination_budget(pool, score, direction, chosen_ids, shown),
                           config=PNG_CONFIG, key="selection_ranking_chart")
    charts[1].plotly_chart(cluster_distribution(selected),
                           config=PNG_CONFIG, key="selection_cluster_chart")
    if not selected.empty:
        candidate = st.selectbox("Candidate details", selected.sequence_id.tolist())
        details = selected[selected.sequence_id.eq(candidate)].iloc[0]
        with st.expander("Scores / Replicates", expanded=False):
            st.code(details.sequence, language=None)
            detail_columns = [score] + catalog.column.tolist() + [column for column in selected if column.startswith("off_target__")]
            score_values = [{"score": column, "value": details[column]} for column in dict.fromkeys(detail_columns) if column in details]
            st.dataframe(pd.DataFrame(score_values), hide_index=True)
        export_config = st.session_state["active_config"].to_dict()
        for field in ["filters", "composites", "selection", "registry", "bootstrap_samples", "random_seed"]:
            export_config[field] = config.to_dict()[field]
        exports = export_selection(selected, export_config)
        buttons = st.columns(3)
        for panel, extension, mime in zip(buttons, ["csv", "fasta", "zip"], ["text/csv", "text/plain", "application/zip"]):
            panel.download_button(extension.upper(), exports[extension], f"selected.{extension}", mime, icon=":material/download:")