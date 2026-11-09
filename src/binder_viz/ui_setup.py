from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from .config import Aggregation, Folder, infer_target
from .experiments import load_experiments
from .loader import discover_files, discover_run_folders, replicate_selector_columns
from .paths import input_files, path_list, yaml_mapping
from .registry import DEFAULT_SCORES, load_registry, save_registry, score_catalog


def local_source(label: str, value: str, key: str, suffixes: tuple[str, ...]) -> str:
    value = st.text_input(label, value, key=key)
    if value.strip() and Path(value).expanduser().is_dir():
        try:
            matches = input_files(value, suffixes)
        except ValueError as error:
            st.warning(str(error))
            return ""
        value = st.selectbox("Discovered file", [str(path) for path in matches], key=f"{key}_file")
    return value


@st.cache_data(show_spinner=False, max_entries=8)
def cached_selector_columns(folders, file_states):
    return replicate_selector_columns(folders)


def aggregation_controls(stage: str, settings: Aggregation, available: list[str], registry: list[dict]) -> Aggregation:
    st.markdown(f"**{stage.capitalize()}**")
    columns = st.columns([1, 1, 1])
    modes = ["best", "worst", "mean", "median", "none"]
    mode = columns[0].selectbox("Aggregation", modes, index=modes.index(settings.mode), key=f"setup_{stage}_mode")
    options = list(available)
    if settings.composite and settings.selector not in options:
        options.append(settings.selector)
    preferred = settings.selector if settings.selector in options else options[0] if options else None
    selector_key, direction_key = f"setup_{stage}_selector", f"setup_{stage}_direction"
    if st.session_state.get(selector_key) not in options:
        st.session_state.pop(selector_key, None)
    known_directions = {row["name"]: row["direction"] for row in registry or DEFAULT_SCORES if row["stage"] == stage}
    known_directions["replicate_composite"] = "higher_better"

    def update_direction():
        chosen = st.session_state.get(selector_key)
        known = known_directions.get(chosen)
        if known in {"higher_better", "lower_better"}:
            st.session_state[direction_key] = known

    selector = columns[1].selectbox("Selector score", options, index=options.index(preferred) if preferred else None,
                                    key=selector_key, disabled=not options, on_change=update_direction,
                                    format_func=lambda value: "Composite (consolidation)" if value == "replicate_composite" else value,
                                    placeholder="No numeric scores detected")
    directions = ["higher_better", "lower_better"]
    initial_direction = known_directions.get(selector, settings.direction)
    direction = columns[2].selectbox("Selector direction", directions,
                                      index=directions.index(initial_direction) if initial_direction in directions else directions.index(settings.direction),
                                      key=direction_key)
    std = st.checkbox("Include replicate standard deviations", settings.std, key=f"setup_{stage}_std")
    override_text = st.text_area("Per-column overrides (YAML or local file/folder)", yaml.safe_dump(settings.overrides, sort_keys=False), height=68, key=f"setup_{stage}_overrides")
    overrides = yaml_mapping(override_text, "overrides")
    if not isinstance(overrides, dict):
        raise ValueError("Aggregation overrides must be a mapping")
    return Aggregation(mode, selector or settings.selector, direction, overrides, std,
                       settings.composite if selector == settings.selector else {})


@st.cache_data(show_spinner=False, max_entries=8)
def experimental_preview(specs, labels, policy, file_states):
    return load_experiments(specs, label_mapping=labels, duplicate_policy=policy)


def experimental_controls(config) -> list[str]:
    st.subheader("Experimental Data")
    paths = st.text_area("Experimental CSV files / local folders", height=90, key="setup_experiment_paths")
    source = st.selectbox("Source for added files", ["auto", "internal", "public"], key="setup_experiment_source")
    if st.button("Load experimental data", icon=":material/database:", disabled=not paths.strip() and not config.experiments):
        specs = {str(Path(spec["path"]).expanduser().resolve()): spec for spec in config.experiments}
        for path in path_list(paths):
            specs[path] = {**specs.get(path, {}), "path": path, "source": source}
        config.experiments = list(specs.values())
        st.session_state.pop("setup_experiments", None)
    with st.expander("Experimental Files and Labels", expanded=bool(config.experiments)):
        specs = [{"path": spec.get("path", ""), "source": spec.get("source", "auto"), "target": spec.get("target", ""),
                  "columns": json.dumps(spec.get("columns", {})), "kd_unit": spec.get("kd_unit", "auto"),
                  "detection_limit_nM": spec.get("detection_limit_nM")} for spec in config.experiments]
        edits = st.data_editor(pd.DataFrame(specs, columns=["path", "source", "target", "columns", "kd_unit", "detection_limit_nM"]),
                               hide_index=True, num_rows="dynamic", key="setup_experiments",
                               column_config={"path": st.column_config.TextColumn("Experimental CSV / local folder", width="large"),
                                              "source": st.column_config.SelectboxColumn("Source", options=["auto", "internal", "public"]),
                                              "columns": st.column_config.TextColumn("Column map (JSON / YAML / local file or folder)"),
                                              "kd_unit": st.column_config.SelectboxColumn("Kd input unit", options=["auto", "pM", "nM", "uM", "mM", "M"]),
                                              "detection_limit_nM": st.column_config.NumberColumn("Detection limit (nM)", min_value=0.0)})
        config.experiments = []
        for row in edits.to_dict("records"):
            if not row.get("path") or pd.isna(row["path"]):
                continue
            spec = {"path": row["path"], "source": row.get("source") or "auto",
                    "columns": yaml_mapping(row["columns"], "columns") if isinstance(row.get("columns"), str) and row["columns"].strip() else {}}
            if isinstance(row.get("target"), str) and row["target"]:
                spec["target"] = row["target"]
            if isinstance(row.get("kd_unit"), str) and row["kd_unit"] != "auto":
                spec["kd_unit"] = row["kd_unit"]
            if pd.notna(row.get("detection_limit_nM")):
                spec["detection_limit_nM"] = float(row["detection_limit_nM"])
            config.experiments.append(spec)
        mapping_text = st.text_area("Class labels (YAML or local file/folder)", yaml.safe_dump(config.label_mapping), height=68, key="setup_label_mapping")
        config.label_mapping = yaml_mapping(mapping_text, "label_mapping")
        policies = ["flag", "aggregate", "error"]
        config.experimental_duplicates = st.selectbox("Repeated measurements", policies, index=policies.index(config.experimental_duplicates), key="setup_experimental_duplicates")
    file_states = []
    for spec in config.experiments:
        try:
            for path in input_files(spec["path"], (".csv",)):
                metadata = path.stat()
                file_states.append((str(path.resolve()), metadata.st_mtime_ns, metadata.st_size))
        except (OSError, ValueError) as error:
            file_states.append((spec["path"], str(error)))
    preview = experimental_preview(config.experiments, config.label_mapping, config.experimental_duplicates, file_states)
    targets = sorted(preview.table.target.dropna().unique())
    st.session_state["setup_available_targets"] = targets
    st.session_state["setup_experimental_preview"] = preview
    if not preview.table.empty:
        counts = st.columns(2)
        counts[0].metric("Experimental measurements", len(preview.table))
        counts[1].metric("Available experimental targets", len(targets))
        with st.expander("Available Experimental Targets"):
            summary = preview.table.groupby(["target", "source"], dropna=False).size().rename("measurements").reset_index()
            st.dataframe(summary, hide_index=True)
    if preview.warnings:
        with st.expander(f"Experimental warnings ({len(preview.warnings)})"):
            for warning in preview.warnings:
                st.warning(warning)
    return targets


def render_setup(config, on_load):
    experimental_targets = experimental_controls(config)
    st.subheader("Run Folders")
    custom_targets = st.session_state.setdefault("setup_custom_targets", [])
    with st.expander("Discover Local Runs"):
        roots = st.text_area("Local run-data paths", height=110, key="setup_run_root",
                             placeholder="/data/batch1\n/data/batch2")
        fields = st.columns(2)
        target_options = sorted(set(experimental_targets) | set(custom_targets) | {folder.target for folder in config.folders})
        choice = fields[0].selectbox("Target for discovered runs", ["Infer from folder", "New target"] + target_options,
                                     key="setup_discovery_target_choice")
        target = "" if choice in {"Infer from folder", "New target"} else choice
        if choice == "New target":
            target = fields[0].text_input("New target name", key="setup_new_target").strip()
            if target and target not in target_options:
                target_options.append(target)
        model = fields[1].text_input("Model for discovered runs (optional)", key="setup_discovery_model")
        if st.button("Add discovered runs", icon=":material/folder_open:", disabled=not roots.strip()):
            if choice == "New target" and not target:
                st.warning("Enter a name for the new target.")
            else:
                discovered = set()
                for root in path_list(roots):
                    runs = discover_run_folders(root)
                    if not runs:
                        st.warning(f"No phase2 or phase3 CSVs found in {root}")
                    discovered.update(path.resolve() for path in runs)
                existing = {folder.path for folder in config.folders}
                added = [Folder(str(path), target or infer_target(path.name, experimental_targets), model)
                         for path in sorted(discovered) if str(path) not in existing]
                config.folders.extend(added)
                target_options = sorted(set(target_options) | {folder.target for folder in config.folders})
                if added:
                    st.session_state["setup_folders_version"] = st.session_state.get("setup_folders_version", 0) + 1
                    st.success(f"Added {len(added)} runs")
    folders = pd.DataFrame([asdict(folder) for folder in config.folders], columns=["path", "target", "model"])
    version = st.session_state.get("setup_folders_version", 0)
    naming = st.columns([3, 1])
    new_target = naming[0].text_input("Add a target name", key="setup_extra_target",
                                      placeholder="Target without experimental data").strip()
    if naming[1].button("Add to target list", icon=":material/add:", disabled=not new_target):
        if new_target not in target_options:
            st.session_state["setup_custom_targets"] = sorted(set(custom_targets) | {new_target})
            target_options = sorted(set(target_options) | {new_target})
        st.rerun()
    with st.form("setup_folder_edits", border=False, enter_to_submit=False):
        edited = st.data_editor(folders, num_rows="dynamic", hide_index=True,
                                key=f"setup_folders_{version}",
                                column_config={"path": st.column_config.TextColumn("Run file / folder", width="large"),
                                               "target": st.column_config.SelectboxColumn("Target", options=target_options, required=True,
                                                                                          help="Pick a known target, or add a new name above to make it selectable."),
                                               "model": "Cofolding model"})
        apply_folders = st.form_submit_button("Apply run-folder edits", icon=":material/check:", key="setup_apply_folders")
    if apply_folders:
        updated = []
        for row in edited.fillna("").to_dict("records"):
            path = str(row.get("path", "")).strip()
            if not path:
                continue
            target = str(row.get("target") or "").strip() or infer_target(Path(path).name, experimental_targets)
            updated.append(Folder(path, target, str(row.get("model") or "")))
        config.folders = updated
        st.session_state["setup_folders_version"] = version + 1
        st.rerun()
    with st.expander("Detected files"):
        detected = [{"target": folder.target, "model": folder.model, "file": str(path), "stage": f"phase{phase}"}
                    for folder in config.folders for path, phase in discover_files(folder.path)]
        st.dataframe(pd.DataFrame(detected), hide_index=True)
        if config.folders:
            grid = pd.DataFrame([asdict(folder) for folder in config.folders]).groupby(["target", "model"]).size().unstack(fill_value=0)
            st.dataframe(grid)
    file_states = []
    for folder in config.folders:
        for path, _ in discover_files(folder.path):
            try:
                metadata = path.stat()
                file_states.append((str(path.resolve()), metadata.st_mtime_ns, metadata.st_size))
            except OSError as error:
                file_states.append((str(path), str(error)))
    selectors, selector_warnings = cached_selector_columns(config.folders, file_states)
    with st.expander("Replicate Handling", expanded=not bool(config.folders)):
        config.include_phase3 = st.checkbox("Include phase3 / Rosetta", config.include_phase3, key="setup_include_phase3")
        config.phase2 = aggregation_controls("phase2", config.phase2, selectors["phase2"], config.registry)
        if config.include_phase3:
            config.phase3 = aggregation_controls("phase3", config.phase3, selectors["phase3"], config.registry)
        config.duplicates = st.selectbox("Duplicate replicate policy", ["first", "last", "error"], index=["first", "last", "error"].index(config.duplicates), key="setup_duplicates")
        if selector_warnings:
            with st.expander("Column discovery warnings"):
                for warning in selector_warnings:
                    st.warning(warning)
    with st.expander("Score Registry"):
        config.registry_path = local_source("Registry file / local folder", config.registry_path, "setup_registry_path", (".csv", ".yaml", ".yml"))
        if st.button("Open registry", icon=":material/folder_open:", disabled=not config.registry_path):
            config.registry = load_registry(config.registry_path)
            st.session_state.pop("setup_registry", None)
        registry = config.registry if config.registry else DEFAULT_SCORES
        entries = st.data_editor(pd.DataFrame(registry), num_rows="dynamic", hide_index=True,
                                 key="setup_registry", column_config={"direction": st.column_config.SelectboxColumn("Direction", options=["higher_better", "lower_better", "unknown"]),
                                                                     "stage": st.column_config.SelectboxColumn("Stage", options=["phase2", "phase3", "derived"])})
        config.registry = entries.where(pd.notna(entries), None).to_dict("records")
        if st.button("Save registry", icon=":material/save:", disabled=not config.registry_path):
            save_registry(config.registry, config.registry_path)
            st.success("Registry saved")
    if st.button("Load", type="primary", icon=":material/database:", disabled=not config.folders):
        on_load(config)
        st.rerun()
    result = st.session_state.get("result")
    if result is None:
        return
    st.subheader("Validation")
    st.dataframe(result.report, hide_index=True)
    if not result.report.empty:
        folder_counts = pd.DataFrame([asdict(folder) for folder in config.folders]).groupby(["target", "model"]).size().rename("folders")
        row_counts = result.replicate_counts.groupby(["target", "model", "stage"])["n_replicates"].sum().unstack(fill_value=0)
        st.dataframe(row_counts.join(folder_counts))
    if not result.replicate_counts.empty:
        import plotly.express as px
        from .charts import layout, PNG_CONFIG

        figure = px.histogram(result.replicate_counts, x="n_replicates", color="stage", barmode="overlay", color_discrete_sequence=["#087f75", "#e17b67"])
        st.plotly_chart(layout(figure, "Replicates per sequence", 280), config=PNG_CONFIG)
    experiments = st.session_state.get("experiments")
    if experiments is not None and not experiments.table.empty:
        total = len(experiments.table)
        st.metric("Experimental match rate", f"{(total - len(experiments.unmatched)) / total:.0%}")
        with st.expander("Unmatched / repeated experimental measurements"):
            st.dataframe(experiments.unmatched, hide_index=True)
            st.dataframe(experiments.duplicates, hide_index=True)
    with st.expander("Score Directions", expanded=False):
        catalog = st.session_state["catalog"]
        edited_catalog = st.data_editor(catalog, hide_index=True, key=f"setup_directions_{result.cache_key}",
                                        disabled=["column", "name", "stage", "model", "inferred", "missing_fraction", "constant"],
                                        column_config={"direction": st.column_config.SelectboxColumn("Direction", options=["unknown", "higher_better", "lower_better"]),
                                                       "confirmed": st.column_config.CheckboxColumn("Confirmed"),
                                                       "inferred": st.column_config.CheckboxColumn("Inferred")})
        if st.button("Apply directions", icon=":material/check:"):
            overrides = edited_catalog.to_dict("records")
            config.registry = [row for row in config.registry if not row.get("column")] + overrides
            st.session_state["catalog"], _ = score_catalog(result.table, config.registry, experiments.table if experiments is not None else None)
            st.rerun()
    issues = result.warnings + (experiments.warnings if experiments is not None else []) + st.session_state.get("registry_warnings", [])
    if issues:
        with st.expander(f"Warnings ({len(issues)})"):
            for warning in dict.fromkeys(issues):
                st.warning(warning)
    st.dataframe(result.table.head(1000), hide_index=True)