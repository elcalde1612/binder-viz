from pathlib import Path
import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from binder_viz.config import Folder, SessionConfig
from binder_viz.demo import create_demo


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_empty_app():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert [tab.label for tab in app.tabs] == ["Setup", "Analysis", "Candidate Selection"]


def test_demo_load_analyze_select(tmp_path):
    config = create_demo(tmp_path / "example")
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = config.to_dict()
    app.session_state["auto_load"] = True
    app.run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["result"].table.sequence_id.nunique() == 48
    assert len(app.session_state["selected_candidates"]) == 12
    button = next(button for button in app.button if button.label == "Analyze")
    button.click().run(timeout=120)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    ranking = app.session_state["analysis_ranking"]
    assert not ranking.empty
    assert ranking.n.max() == 32
    assert ranking.value.max() > 0.8
    assert app.session_state["selected_candidates"].sequence_id.nunique() == 12


def test_specificity_controls_keep_hidden_off_target_evidence(tmp_path):
    folders = []
    for target, sequences, scores in [("A", ["AAAA", "CCCC", "DDDD"], [0.95, 0.9, 0.8]),
                                       ("B", ["AAAA", "CCCC"], [0.85, 0.2])]:
        stats = tmp_path / target / "traj_stats"
        stats.mkdir(parents=True)
        pd.DataFrame({"sequence": sequences, "iptm": scores, "pdbname": [f"design_{index}" for index in range(len(scores))]}).to_csv(stats / "test_phase2.csv", index=False)
        folders.append(Folder(str(stats.parent), target, "af2"))
    config = SessionConfig(folders=folders, include_phase3=False, filters={"targets": ["A"]},
                           selection={"positive_targets": ["A"], "negative_selection": [
                               {"target": "B", "score": "iptm__af2", "direction": "higher_better", "threshold": 0.7}],
                               "cluster": False, "exclude_tested": False})
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = config.to_dict()
    app.session_state["auto_load"] = True
    app.run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    selected = app.session_state["selected_candidates"]
    assert selected.sequence.tolist() == ["CCCC"]
    assert selected.off_target__B__iptm__af2.tolist() == [0.2]
    assert "B" in app.multiselect(key="selection_negative_targets").options
    assert app.session_state["config"].selection["negative_selection"][0]["threshold"] == 0.7
    app.checkbox(key="selection_exclude_missing").uncheck().run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["selected_candidates"].sequence.tolist() == ["CCCC", "DDDD"]


def test_session_local_folder_file_choice(tmp_path):
    (tmp_path / "first.yaml").write_text(SessionConfig(random_seed=123).to_yaml())
    (tmp_path / "second.yaml").write_text(SessionConfig(random_seed=456).to_yaml())
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.text_input(key="session_path").set_value(str(tmp_path)).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="session_path_file").options == [str(tmp_path / "first.yaml"), str(tmp_path / "second.yaml")]
    app.selectbox(key="session_path_file").set_value(str(tmp_path / "second.yaml")).run(timeout=30)
    next(button for button in app.button if button.label == "Open").click().run(timeout=30)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["config"].random_seed == 456


def write_setup_run(root, name):
    stats = root / name / "traj_stats"
    stats.mkdir(parents=True)
    pd.DataFrame({"pdbname": [f"{name}_1", f"{name}_2"], "sequence": ["AAAA", "CCCC"],
                  "iptm": [0.9, 0.7]}).to_csv(stats / "test_phase2.csv", index=False)
    return stats.parent


def test_experiments_first_and_bulk_run_assignment(tmp_path):
    experiments = tmp_path / "binding_dataset.csv"
    pd.DataFrame({"target": ["Known-A", "Known-B"], "sequence": ["EEEE", "FFFF"],
                  "binding_strength": ["Strong", "None"], "data_origin": ["internal", "proteinbase"]}).to_csv(experiments, index=False)
    first = write_setup_run(tmp_path, "batch1_af2")
    second = write_setup_run(tmp_path, "batch2_af2")
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert app.tabs[0].subheader[0].value == "Experimental Data"
    app.text_area(key="setup_experiment_paths").set_value(str(experiments)).run(timeout=30)
    next(button for button in app.button if button.label == "Load experimental data").click().run(timeout=30)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["setup_available_targets"] == ["Known-A", "Known-B"]
    assert app.session_state["config"].folders == []
    assert "Known-A" in app.selectbox(key="setup_discovery_target_choice").options
    app.selectbox(key="setup_discovery_target_choice").set_value("Known-A")
    app.text_area(key="setup_run_root").set_value(json.dumps([str(first), str(second), str(tmp_path)])).run(timeout=30)
    next(button for button in app.button if button.label == "Add discovered runs").click().run(timeout=30)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    folders = app.session_state["config"].folders
    assert len(folders) == 2
    assert {folder.target for folder in folders} == {"Known-A"}
    assert {folder.model for folder in folders} == {"af2"}
    app.checkbox(key="setup_include_phase3").uncheck().run(timeout=30)
    next(button for button in app.button if button.label == "Load").click().run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert set(app.session_state["selected_candidates"].target) == {"Known-A"}


def test_new_target_without_experimental_data(tmp_path):
    first = write_setup_run(tmp_path, "new_batch1_af2")
    second = write_setup_run(tmp_path, "new_batch2_af2")
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.selectbox(key="setup_discovery_target_choice").set_value("New target").run(timeout=30)
    app.text_input(key="setup_new_target").set_value("Novel-X")
    app.text_area(key="setup_run_root").set_value(f"{first}\n{second}").run(timeout=30)
    next(button for button in app.button if button.label == "Add discovered runs").click().run(timeout=30)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["setup_available_targets"] == []
    config = app.session_state["config"]
    assert len(config.folders) == 2
    assert {folder.target for folder in config.folders} == {"Novel-X"}
    assert not config.experiments
    app.checkbox(key="setup_include_phase3").uncheck().run(timeout=30)
    next(button for button in app.button if button.label == "Load").click().run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert not app.session_state["selected_candidates"].empty
    assert set(app.session_state["selected_candidates"].target) == {"Novel-X"}
    assert app.selectbox(key="analysis_explore_score").value == "composite__af2"
    assert len(app.get("plotly_chart")) >= 5
    app.selectbox(key="analysis_explore_score").set_value("iptm__af2").run(timeout=60)
    app.toggle(key="analysis_explore_replicates").set_value(True).run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]


def test_discovery_matches_experimental_targets_and_keeps_ambiguous_names(tmp_path):
    measurements = tmp_path / "experimental.csv"
    pd.DataFrame({"target": ["MM2", "Epidermal Growth Factor Receptor", "Kinase Alpha", "Kinase Alphb"],
                  "sequence": ["AAAA", "CCCC", "DDDD", "EEEE"], "binding": [True] * 4}).to_csv(measurements, index=False)
    first = write_setup_run(tmp_path, "mm2_binders_af2")
    second = write_setup_run(tmp_path, "epiderml_growth_factor_receptor_opendde")
    third = write_setup_run(tmp_path, "kinase_alphc_af2")
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = SessionConfig(experiments=[{"path": str(measurements)}]).to_dict()
    app.run(timeout=30)
    app.text_area(key="setup_run_root").set_value(f"{first}\n{second}\n{third}").run(timeout=30)
    next(button for button in app.button if button.label == "Add discovered runs").click().run(timeout=30)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    targets = {Path(folder.path).name: folder.target for folder in app.session_state["config"].folders}
    assert targets == {first.name: "MM2", second.name: "Epidermal Growth Factor Receptor", third.name: third.name}
    app.checkbox(key="setup_include_phase3").uncheck().run(timeout=30)
    assert {Path(folder.path).name: folder.target for folder in app.session_state["config"].folders} == targets


def test_run_folder_edits_are_batched_and_persist_after_apply(tmp_path):
    def apply_edits(editor_key, edits):
        app.button(key="setup_apply_folders").click()
        states = app._tree.get_widget_states()
        editor = next(element for element in app.dataframe if element.proto.id.endswith(editor_key))
        state = next((state for state in states.widgets if state.id == editor.proto.id), None)
        if state is None:
            state = states.widgets.add(id=editor.proto.id)
        state.string_value = json.dumps(edits)
        app._run(widget_state=states, timeout=30)

    first = write_setup_run(tmp_path, "one_af2")
    second = write_setup_run(tmp_path, "two_af2")
    config = SessionConfig(folders=[Folder(str(first), "Target-A", "af2"), Folder(str(second), "Target-B", "af2")])
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = config.to_dict()
    app.run(timeout=30)
    app.session_state["setup_folders_0"] = {"edited_rows": {0: {"target": "Target-B", "model": "boltz"}},
                                            "added_rows": [], "deleted_rows": []}
    app.run(timeout=30)
    assert app.session_state["config"].folders[0].target == "Target-A"
    edits = {"edited_rows": {0: {"target": "Target-B", "model": "boltz"},
                              1: {"target": "Target-A", "model": "opendde"}},
             "added_rows": [], "deleted_rows": []}
    app.session_state["setup_folders_0"] = edits
    app.run(timeout=30)
    assert [folder.model for folder in app.session_state["config"].folders] == ["af2", "af2"]
    apply_edits("setup_folders_0", edits)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert [(folder.target, folder.model) for folder in app.session_state["config"].folders] == [("Target-B", "boltz"), ("Target-A", "opendde")]
    app.checkbox(key="setup_include_phase3").uncheck().run(timeout=30)
    assert [(folder.target, folder.model) for folder in app.session_state["config"].folders] == [("Target-B", "boltz"), ("Target-A", "opendde")]
    apply_edits("setup_folders_1", {"edited_rows": {0: {"model": "af2"}}, "added_rows": [], "deleted_rows": [1]})
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert [(folder.target, folder.model) for folder in app.session_state["config"].folders] == [("Target-B", "af2")]


def test_exploration_skips_empty_default_composite(tmp_path):
    stats = tmp_path / "custom_af2" / "traj_stats"
    stats.mkdir(parents=True)
    pd.DataFrame({"sequence": ["AAAA", "CCCC"], "custom_numeric": [1.0, 2.0],
                  "pdbname": ["a", "c"]}).to_csv(stats / "test_phase2.csv", index=False)
    config = SessionConfig(folders=[Folder(str(stats.parent), "T", "af2")], include_phase3=False)
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = config.to_dict()
    app.session_state["auto_load"] = True
    app.run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.selectbox(key="analysis_explore_score").value == "custom_numeric__af2"
    assert "composite__af2" not in app.selectbox(key="analysis_explore_score").options


def test_chart_layout_inherits_active_theme():
    import plotly.graph_objects as go

    from binder_viz.charts import layout

    figure = layout(go.Figure(), "Theme-aware chart")
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.font.color is None


def test_replicate_selector_dropdown_and_composite_load(tmp_path):
    stats = tmp_path / "run_af2" / "traj_stats"
    stats.mkdir(parents=True)
    pd.DataFrame({"pdbname": ["a_seed_1", "a_seed_2", "c_seed_1"], "sequence": ["AAAA", "AAAA", "CCCC"],
                  "iptm": [0.9, 0.7, 0.5], "ipsae": [0.1, 0.9, 0.4],
                  "apbs_dG_elec": [-1, -10, -5], "custom_numeric": [1, 2, 3]}).to_csv(stats / "test_phase2.csv", index=False)
    config = SessionConfig(folders=[Folder(str(stats.parent), "T", "af2")], include_phase3=False)
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = config.to_dict()
    app.run(timeout=30)
    assert not app.exception
    selector = app.selectbox(key="setup_phase2_selector")
    assert {"iptm", "ipsae", "apbs_dG_elec", "custom_numeric", "Composite (consolidation)"} <= set(selector.options)
    selector.set_value("replicate_composite").run(timeout=30)
    next(button for button in app.button if button.label == "Load").click().run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.session_state["config"].phase2.selector == "replicate_composite"
    result = app.session_state["result"].table.set_index("sequence")
    assert result.loc["AAAA", "iptm__af2"] == 0.7
    assert "composite__af2" in result


def test_exploratory_figures_keep_measurements_and_candidates_distinct():
    from binder_viz.charts import affinity_marker_sizes, candidate_nomination_budget, cluster_distribution, kd_scatter, method_agreement, score_distribution, selected_method_blocks, stratified_distribution

    table = pd.DataFrame({"sequence_id": ["a", "b", "c"], "target": ["T"] * 3,
                          "composite__af2": [90, 60, 30], "composite__opendde": [85, 55, 20],
                          "score_refold__af2": [95, 50, 20], "binding_class": ["strong", "none", None],
                          "binds": pd.array([True, False, pd.NA], dtype="boolean"),
                          "source": ["internal", "public", None], "kd_nM": [10, 0.1, None],
                          "kd_qualifier": ["=", "public_not_applicable", "missing"], "cluster_id": [1, 2, 3]})
    figure = method_agreement(table, "composite__af2", "composite__opendde", ["c"])
    assert any(trace.name == "unmeasured" and trace.marker.symbol == "diamond" for trace in figure.data)
    assert any(trace.name == "Selected candidates" for trace in figure.data)
    kd = kd_scatter(table, "composite__af2")
    assert sum(len(trace.x) for trace in kd.data) == 1
    assert kd.layout.yaxis.type == "log"
    assert affinity_marker_sizes(pd.Series([1, 100, None])).tolist() == [18, 8, 8]
    figures = [score_distribution(table, table.iloc[:2], "composite__af2", ["a"]),
               stratified_distribution(table, "composite__af2", "binding_class"),
               selected_method_blocks(table, ["a"], ["af2", "opendde"]),
               candidate_nomination_budget(table, "composite__af2", "higher_better", ["a"]),
               cluster_distribution(table)]
    assert all(figure.data for figure in figures)
    assert all(figure.layout.paper_bgcolor == "rgba(0,0,0,0)" for figure in figures)
    assert kd_scatter(table.iloc[:0], "composite__af2").layout.annotations
    repeats = pd.concat([table, table.iloc[:1]], ignore_index=True)
    assert len(score_distribution(repeats, repeats, "composite__af2").data[0].x) == 3
    assert len(score_distribution(repeats, repeats, "composite__af2", replicate_rows=True).data[0].x) == 4


def test_model_agreement_independent_axes_colors_and_experimental_names(tmp_path):
    stats = tmp_path / "run_af2" / "traj_stats"
    stats.mkdir(parents=True)
    pd.DataFrame({"pdbname": ["a", "c", "d"], "sequence": ["AAAA", "CCCC", "DDDD"],
                  "iptm": [0.9, 0.7, 0.3], "ipsae": [0.8, 0.6, 0.2]}).to_csv(stats / "test_phase2.csv", index=False)
    measurements = tmp_path / "experimental.csv"
    measurements.write_text("target,sequence,id,name,binding_strength,binding\nT,AAAA,001,Named A,Strong,True\nT,CCCC,002,Named C,,False\n")
    config = SessionConfig(folders=[Folder(str(stats.parent), "T", "af2")], include_phase3=False,
                           experiments=[{"path": str(measurements)}])
    app = AppTest.from_file(str(APP))
    app.session_state["pending_config"] = config.to_dict()
    app.session_state["auto_load"] = True
    app.run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert "Named A" in app.multiselect(key="analysis_explore_highlights").options
    assert "Named C" in app.multiselect(key="analysis_explore_highlights").options
    app.selectbox(key="analysis_agreement_x").set_value("ipsae__af2")
    app.selectbox(key="analysis_agreement_y").set_value("iptm__af2")
    app.radio(key="analysis_agreement_color").set_value("binds").run(timeout=60)
    assert not app.exception
    assert not app.error, [item.value for item in app.error]
    assert app.selectbox(key="analysis_explore_score").value == "composite__af2"
    chart = next(element for element in app.get("plotly_chart") if element.proto.id.endswith("analysis_explore_agreement"))
    figure = json.loads(chart.proto.spec)
    assert figure["layout"]["xaxis"]["title"]["text"] == "ipsae__af2"
    assert figure["layout"]["yaxis"]["title"]["text"] == "iptm__af2"
    assert {trace["name"] for trace in figure["data"]} >= {"binding", "nonbinding", "unmeasured"}
    assert "Named A" in [name for trace in figure["data"] for name in trace.get("text", [])]
    app.selectbox(key="analysis_agreement_y").set_value("ipsae__af2")
    app.radio(key="analysis_agreement_color").set_value("binding_class").run(timeout=60)
    assert not app.exception
    chart = next(element for element in app.get("plotly_chart") if element.proto.id.endswith("analysis_explore_agreement"))
    assert "strong" in {trace["name"] for trace in json.loads(chart.proto.spec)["data"]}