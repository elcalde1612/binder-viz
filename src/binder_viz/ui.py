from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from .config import SessionConfig
from .demo import create_demo
from .experiments import load_experiments
from .loader import content_key, load_predictions
from .paths import output_file
from .registry import DEFAULT_SCORES, load_registry, score_catalog
from .ui_analysis import derived_scores, render_analysis
from .ui_selection import render_selection
from .ui_setup import local_source, render_setup


ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load(config_yaml: str, signature: str):
    return load_predictions(SessionConfig.from_yaml(config_yaml), ROOT / ".cache" / "predictions")


def load_session(config: SessionConfig):
    with st.spinner("Consolidating predictions"):
        if config.registry_path and not config.registry:
            config.registry = load_registry(config.registry_path)
        result = cached_load(config.to_yaml(), content_key(config))
        if result.table.empty:
            st.error("No usable prediction rows. Check detected files and warnings.")
            for warning in result.warnings:
                st.warning(warning)
            return
        experiments = load_experiments(config.experiments, result.table, config.label_mapping, config.experimental_duplicates)
        catalog, issues = score_catalog(result.table, config.registry or DEFAULT_SCORES, experiments.table)
    st.session_state.update(result=result, experiments=experiments, catalog=catalog, registry_warnings=issues,
                            active_config=SessionConfig.from_dict(config.to_dict()),
                            dataset_key=sha256((result.cache_key + experiments.table.to_json()).encode()).hexdigest())
    st.session_state.pop("analysis_ranking", None)
    st.session_state.pop("best_score", None)


def sidebar(config):
    st.sidebar.header("Session")
    with st.sidebar:
        path = local_source("Session YAML file / local folder", str(ROOT / "session.yaml"), "session_path", (".yaml", ".yml"))
    open_column, save_column = st.sidebar.columns(2)
    if open_column.button("Open", icon=":material/folder_open:", disabled=not path):
        st.session_state["pending_config"] = SessionConfig.from_yaml(Path(path).expanduser().read_text()).to_dict()
        st.rerun()
    if save_column.button("Save", icon=":material/save:"):
        output_file(st.session_state["session_path"], "session.yaml").write_text(config.to_yaml())
        st.sidebar.success("Session saved")
    upload = st.sidebar.file_uploader("Import session", type=["yml", "yaml"], key="session_upload")
    if upload is not None and st.sidebar.button("Apply imported session", icon=":material/upload:"):
        st.session_state["pending_config"] = SessionConfig.from_yaml(upload.getvalue().decode()).to_dict()
        st.rerun()
    st.sidebar.download_button("Session YAML", config.to_yaml(), "session.yaml", "application/yaml", icon=":material/download:")
    if st.sidebar.button("Load example", icon=":material/science:"):
        st.session_state["pending_config"] = create_demo(ROOT / ".cache" / "example").to_dict()
        st.session_state["auto_load"] = True
        st.rerun()


def global_filters(table, catalog, config):
    st.sidebar.header("Global Filters")
    filters = config.filters
    targets = sorted(table.target.unique())
    chosen = st.sidebar.multiselect("Targets", targets, default=[value for value in filters.get("targets", targets) if value in targets])
    models = sorted({folder.model for folder in config.folders})
    chosen_models = st.sidebar.multiselect("Models", models, default=[value for value in filters.get("models", models) if value in models])
    minimum = st.sidebar.number_input("Minimum pLDDT", 0.0, 100.0, float(filters.get("min_plddt", 0.0)), 1.0)
    config.filters = {"targets": chosen, "models": chosen_models, "min_plddt": minimum}
    selected_catalog = catalog[catalog.model.isin(chosen_models) | catalog.stage.eq("derived")]
    table = table[table.target.isin(chosen)].copy()
    counts = [f"n_rep_p2__{model}" for model in chosen_models if f"n_rep_p2__{model}" in table]
    counts += [f"n_rep_p3__{model}" for model in chosen_models if f"n_rep_p3__{model}" in table]
    if not chosen_models:
        table = table.iloc[:0]
    elif counts:
        table = table[table[counts].fillna(0).sum(axis=1).gt(0)]
    plddt = [f"plddt__{model}" for model in chosen_models if f"plddt__{model}" in table]
    if minimum and plddt:
        table = table[table[plddt].min(axis=1).ge(minimum)]
    elif minimum:
        st.sidebar.warning("No pLDDT scores are available for this filter.")
    return table, selected_catalog


def run():
    st.set_page_config(page_title="binder-viz", page_icon=":material/biotech:", layout="wide")
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap');
    html, body, [class*="css"], .stApp {font-family:'IBM Plex Sans','Helvetica Neue',sans-serif;letter-spacing:0;}
    .stApp {background-image:linear-gradient(90deg,color-mix(in srgb,var(--primary-color) 3%,transparent) 1px,transparent 1px);background-size:32px 32px;}
    h1 {font-family:'IBM Plex Serif',Georgia,serif!important;font-size:2.4rem!important;letter-spacing:0!important;}
    h2 {font-size:1.3rem!important;letter-spacing:0!important;}
    h3 {font-size:1.05rem!important;letter-spacing:0!important;}
    .block-container {padding-top:2rem;padding-bottom:3rem;max-width:1600px;}
    [data-testid="stMetricValue"] {font-size:1.7rem;}
    button {border-radius:5px!important;white-space:normal!important;}
    [data-testid="stSidebar"] {border-right:1px solid color-mix(in srgb,var(--text-color) 12%,transparent);}
    </style>""", unsafe_allow_html=True)
    if "pending_config" in st.session_state:
        data = st.session_state.pop("pending_config")
        for key in list(st.session_state):
            if key.startswith(("setup_", "selection_", "analysis_")) or key in {"result", "experiments", "catalog", "best_score", "active_config"}:
                del st.session_state[key]
        st.session_state["config"] = SessionConfig.from_dict(data)
    if "config" not in st.session_state:
        st.session_state["config"] = SessionConfig(registry=DEFAULT_SCORES.copy())
    config = st.session_state["config"]
    st.title("binder-viz")
    try:
        if st.session_state.pop("auto_load", False):
            load_session(config)
        sidebar(config)
        result = st.session_state.get("result")
        if result is not None:
            summaries = st.columns(4)
            summaries[0].metric("Sequences", result.table.sequence_id.nunique())
            summaries[1].metric("Targets", result.table.target.nunique())
            summaries[2].metric("Models", len({folder.model for folder in st.session_state["active_config"].folders}))
            summaries[3].metric("Replicate rows", f"{result.replicate_counts.n_replicates.sum():,}")
        setup, analysis, selection = st.tabs(["Setup", "Analysis", "Candidate Selection"])
        with setup:
            render_setup(config, load_session)
        result = st.session_state.get("result")
        if result is None:
            with analysis:
                st.info("No predictions loaded.")
            with selection:
                st.info("No predictions loaded.")
            return
        active = st.session_state["active_config"]
        if any(config.to_dict()[field] != active.to_dict()[field] for field in ["folders", "phase2", "phase3", "include_phase3", "duplicates", "experiments", "label_mapping", "experimental_duplicates"]):
            st.warning("Setup has changed. Load to replace the current dataset.")
        experiments = st.session_state["experiments"]
        table, catalog, coefficients, issues = derived_scores(result.table, st.session_state["catalog"], config, experiments.table)
        for issue in issues:
            st.warning(issue)
        off_target_predictions = table
        table, catalog = global_filters(table, catalog, config)
        with analysis:
            if not table.empty:
                render_analysis(table, catalog, config, experiments, coefficients, full_table=off_target_predictions)
            else:
                st.info("No sequences pass the global filters.")
        with selection:
            render_selection(table, catalog, config, experiments, off_target_predictions)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Configuration error: {error}")