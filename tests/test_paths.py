import pandas as pd
import pytest

from binder_viz.config import Aggregation, Folder, SessionConfig, infer_target
from binder_viz.experiments import load_experiments
from binder_viz.loader import discover_files, discover_run_folders, load_predictions, sequence_id
from binder_viz.paths import input_file, input_files, output_file, path_list, yaml_mapping
from binder_viz.registry import load_registry, save_registry


def test_recursive_file_discovery_and_ambiguous_single_input(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = nested / "first.csv"
    first.write_text("sequence,binding_class\nAAAA,strong\n")
    assert input_files(tmp_path, (".csv",)) == [first]
    assert input_file(tmp_path, (".csv",)) == first
    (tmp_path / "second.csv").write_text("sequence,binding_class\nCCCC,none\n")
    with pytest.raises(ValueError, match="choose a specific file"):
        input_file(tmp_path, (".csv",))
    with pytest.raises(ValueError, match="No .yaml files"):
        input_files(tmp_path, (".yaml",))
    with pytest.raises(ValueError, match="does not exist"):
        input_files(tmp_path / "missing", (".csv",))


def test_run_parent_folder_and_direct_csv_discovery(tmp_path):
    for name in ("batch1_af2", "batch2_af2"):
        stats = tmp_path / name / "traj_stats"
        stats.mkdir(parents=True)
        pd.DataFrame({"pdbname": [name], "sequence": ["AAAA"], "iptm": [0.8]}).to_csv(stats / "test_phase2.csv", index=False)
    assert len(discover_files(tmp_path)) == 2
    runs = discover_run_folders(tmp_path)
    assert len(runs) == 2
    direct = runs[0] / "traj_stats" / "test_phase2.csv"
    assert discover_files(direct) == [(direct, 2)]
    config = SessionConfig(folders=[Folder(str(tmp_path), "T", "af2")], include_phase3=False, phase2=Aggregation(mode="mean"))
    result = load_predictions(config)
    assert result.table.n_rep_p2.iloc[0] == 2


def test_experimental_directory_preserves_mapping_and_resolves_sequences(tmp_path):
    predictions = pd.DataFrame({"sequence_id": [sequence_id("T", "AAAA"), sequence_id("T", "CCCC")],
                                "target": ["T", "T"], "sequence": ["AAAA", "CCCC"]})
    (tmp_path / "first.csv").write_text(f"sequence_id,label\n{predictions.sequence_id.iloc[0]},strong\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second.csv").write_text("sequence,label\nCCCC,no\n")
    (nested / "bad.csv").write_text("not_a_sequence,label\ninvalid,strong\n")
    result = load_experiments([{"path": str(tmp_path), "target": "T", "columns": {"binding_class": "label"}}], predictions)
    assert len(result.table) == 2
    assert set(result.table.sequence) == {"AAAA", "CCCC"}
    assert set(result.table.target) == {"T"}
    assert result.table.binds.tolist() == [True, False]
    assert result.unmatched.empty
    assert len(result.warnings) == 1


def test_registry_and_yaml_mapping_folder_inputs(tmp_path):
    rows = [{"name": "iptm", "stage": "phase2", "direction": "higher_better"}]
    save_registry(rows, tmp_path)
    assert load_registry(tmp_path) == rows
    assert output_file(tmp_path, "session.yaml") == tmp_path / "session.yaml"
    mapping = tmp_path / "labels"
    mapping.mkdir()
    (mapping / "labels.yaml").write_text("label_mapping:\n  negative: none\n")
    assert yaml_mapping(str(mapping), "label_mapping") == {"negative": "none"}
    assert yaml_mapping("iptm: mean") == {"iptm": "mean"}
    assert yaml_mapping("{}") == {}
    (tmp_path / "columns.json").write_text('{"columns": {"sequence": "binder_sequence"}}')
    assert yaml_mapping(str(tmp_path / "columns.json"), "columns") == {"sequence": "binder_sequence"}


def test_experiments_load_before_predictions(tmp_path):
    path = tmp_path / "measurements.csv"
    path.write_text("target,sequence,binding_class\nTarget-A,AAAA,strong\nTarget-B,CCCC,none\n")
    result = load_experiments([{"path": str(path)}])
    assert sorted(result.table.target.unique()) == ["Target-A", "Target-B"]
    assert result.table.binds.tolist() == [True, False]
    assert not result.warnings
    assert result.table.sequence_id.tolist() == [sequence_id("Target-A", "AAAA"), sequence_id("Target-B", "CCCC")]


@pytest.mark.parametrize("text", ["/data/run1\n/data/run2\n/data/run1", '["/data/run1", "/data/run2"]',
                                  "- /data/run1\n- /data/run2", "/data/run1, /data/run2"])
def test_multiple_run_path_formats(text):
    assert path_list(text) == ["/data/run1", "/data/run2"]


def test_path_list_preserves_spaces_and_rejects_non_paths():
    assert path_list('"/data/run one", "/data/run,two"') == ["/data/run one", "/data/run,two"]
    assert path_list("") == []
    with pytest.raises(ValueError, match="strings"):
        path_list('["/data/run1", 123]')
    with pytest.raises(ValueError, match="Invalid path list"):
        path_list('["/data/run1"')


def test_mixed_origin_dataset_schema_and_molar_kd(tmp_path):
    path = tmp_path / "binding_dataset.csv"
    pd.DataFrame({"data_origin": ["proteinbase", "internal", "internal", "internal"],
                  "target": ["Public-A", "Internal-B", "Internal-B", "New-C"],
                  "sequence": ["AAAA", "CCCC", "DDDD", "EEEE"],
                  "binding_strength": ["Strong", "Medium", "", ""], "binding": [True, True, False, True],
                  "KD (M)": [1e-9, 2e-8, "", ""]}).to_csv(path, index=False)
    result = load_experiments([{"path": str(path)}])
    assert result.table.source.tolist() == ["public", "internal", "internal", "internal"]
    assert result.table.binding_class.iloc[:3].tolist() == ["strong", "medium", "none"]
    assert result.table.binds.tolist() == [True, True, False, True]
    assert pd.isna(result.table.binding_ord.iloc[3])
    assert pd.isna(result.table.kd_nM.iloc[0])
    assert result.table.kd_nM.iloc[1] == pytest.approx(20)
    assert not result.warnings


@pytest.mark.parametrize("folder,targets,expected", [
    ("MM2_binders_af2", ["MM2", "EGFR"], "MM2"),
    ("hinsr_esmfold2", ["hINSr", "hIDI2"], "hINSr"),
    ("PD-L1_opendde", ["PDL1", "PDGFR"], "PDL1"),
    ("batch12_target_a_af2", ["Target-A", "Target-B"], "Target-A"),
    ("epiderml_growth_factor_receptor_af2", ["Epidermal Growth Factor Receptor", "Insulin Receptor"], "Epidermal Growth Factor Receptor"),
    ("DER21m_af2", ["DER21", "DER21m"], "DER21m"),
    ("DER21_af2", ["DER7"], "DER21_af2"),
    ("FGFR2_af2", ["FGFR1"], "FGFR2_af2"),
    ("EGFR_af2", ["FGFR1"], "EGFR_af2"),
    ("kinase_alphc_af2", ["Kinase Alpha", "Kinase Alphb"], "kinase_alphc_af2"),
    ("pdl1_af2", ["PD-L1", "PDL1"], "pdl1_af2"),
    ("EGFR_HER2_af2", ["EGFR", "HER2"], "EGFR_HER2_af2"),
    ("unknown_af2", ["MM2", "EGFR"], "unknown_af2"),
    ("MM2_binders_af2", [], "MM2_binders_af2"),
])
def test_target_inference_matches_experimental_spelling_conservatively(folder, targets, expected):
    assert infer_target(folder, targets) == expected