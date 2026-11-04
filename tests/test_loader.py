from pathlib import Path

import pandas as pd
import pytest

from binder_viz.config import Aggregation, Folder, SessionConfig
from binder_viz.loader import aggregate_stage, load_predictions, normalize_name, replicate_selector_columns, sequence_id


def write_run(root: Path, phase: int, rows: list[dict], stem="batch"):
    path = root / "traj_stats"
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / f"{stem}_phase{phase}_stats.csv", index=False)


def test_normalization():
    assert normalize_name("binder_seed_1_0001.pdb") == "binder_seed_1"
    assert normalize_name("binder_seed_1") == "binder_seed_1"
    assert normalize_name(None) == ""
    assert sequence_id("T", " aa a ") == sequence_id("T", "AAA")
    assert sequence_id("T", "AAA") != sequence_id("Other", "AAA")


@pytest.mark.parametrize("mode,energy", [("best", -1), ("worst", -99)])
def test_coherent_rows(mode, energy):
    frame = pd.DataFrame({"target": ["T"] * 2, "model": ["M"] * 2, "sequence": ["AAA"] * 2,
                          "iptm": [0.9, 0.1], "energy": [-1, -99]})
    out = aggregate_stage(frame, Aggregation(mode=mode))
    assert out.iloc[0]["energy"] == energy
    assert out.iloc[0]["n_replicates"] == 2
    assert out.iloc[0]["iptm_std"] > 0


def test_multifolder_outer_join_recovery_and_cache(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    write_run(first, 2, [{"sequence": "AAA", "pdbname": "a_seed_1", "iptm": 0.9},
                         {"sequence": "CCC", "pdbname": "c_seed_1", "iptm": 0.2}])
    write_run(second, 2, [{"sequence": "AAA", "pdbname": "a_seed_2", "iptm": 0.1}])
    write_run(first, 3, [{"description": "a_seed_1_0001", "total_score": -10},
                         {"description": "unmatched_0001", "sequence": "DDD", "total_score": -20}])
    config = SessionConfig(folders=[Folder(str(first), "T", "af2"), Folder(str(second), "T", "af2")])
    result = load_predictions(config, tmp_path / "cache")
    assert set(result.table.sequence) == {"AAA", "CCC", "DDD"}
    row = result.table.set_index("sequence").loc["AAA"]
    assert row["iptm__af2"] == 0.9
    assert row["n_rep_p2"] == 2
    assert row["ros_total_score__af2"] == -10
    assert result.report.iloc[0]["matched"] == 1
    assert result.report.iloc[0]["unmatched_phase2"] == 1
    assert result.report.iloc[0]["unmatched_phase3"] == 1
    pd.testing.assert_frame_equal(result.table, load_predictions(config, tmp_path / "cache").table)


def test_duplicates_policy_and_cross_model_identity(tmp_path):
    write_run(tmp_path, 2, [{"sequence": "AAA", "pdbname": "a_seed_1", "iptm": 0.1},
                           {"sequence": "AAA", "pdbname": "a_seed_1", "iptm": 0.9}])
    config = SessionConfig(folders=[Folder(str(tmp_path), "T", "af2")], duplicates="error")
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(config)
    config.duplicates = "last"
    config.folders.append(Folder(str(tmp_path), "T", "opendde"))
    result = load_predictions(config)
    assert len(result.table) == 1
    assert result.table.iloc[0]["iptm__af2"] == result.table.iloc[0]["iptm__opendde"] == 0.9


def test_replicate_join_and_fallback(tmp_path):
    write_run(tmp_path, 2, [{"sequence": "AAA", "pdbname": f"a_seed_{seed}", "iptm": seed / 10} for seed in [1, 2]])
    write_run(tmp_path, 3, [{"sequence": "AAA", "description": f"a_seed_{seed}_0001", "total_score": -seed} for seed in [1, 3]])
    config = SessionConfig(folders=[Folder(str(tmp_path), "T", "af2")], phase2=Aggregation(mode="none"), phase3=Aggregation(mode="none"))
    result = load_predictions(config)
    assert len(result.table) == 3
    assert result.report.iloc[0]["matched"] == 1
    write_run(tmp_path, 2, [{"sequence": "AAA", "pdbname": "a", "iptm": 0.5}])
    result = load_predictions(config)
    assert len(result.table) == 1
    assert any("falling back" in warning for warning in result.warnings)


def test_overrides_and_bad_file(tmp_path):
    write_run(tmp_path, 2, [{"sequence": "AAA", "iptm": 0.9, "plddt": 90}, {"sequence": "AAA", "iptm": 0.1, "plddt": 50}])
    config = SessionConfig(folders=[Folder(str(tmp_path), "T", "af2")], phase2=Aggregation(overrides={"plddt": "mean"}))
    result = load_predictions(config)
    assert result.table.iloc[0]["iptm__af2"] == 0.9
    assert result.table.iloc[0]["plddt__af2"] == 70
    assert SessionConfig.from_yaml(config.to_yaml()).to_dict() == config.to_dict()


@pytest.mark.parametrize("mode,expected", [("best", 60), ("worst", 99)])
def test_composite_selector_keeps_coherent_replicate(mode, expected):
    frame = pd.DataFrame({"target": ["T"] * 3, "model": ["af2"] * 3, "sequence": ["AAA"] * 3,
                          "iptm": [0.95, 0.8, 1.0], "ipsae": [0.1, 0.9, None], "plddt": [99, 60, 100]})
    settings = Aggregation(mode=mode, selector="composite_confidence", composite={
        "weights": {"iptm": 1.0, "ipsae": 3.0},
        "directions": {"iptm": "higher_better", "ipsae": "higher_better"}, "normalization": "rank"})
    result = aggregate_stage(frame, settings)
    assert result.plddt.iloc[0] == expected
    assert result.composite_confidence.notna().all()
    assert result.n_replicates.iloc[0] == 3


def test_composite_selector_export_and_config_roundtrip(tmp_path):
    write_run(tmp_path, 2, [{"sequence": "AAA", "pdbname": "a_seed_1", "iptm": 0.9, "ipsae": 0.1, "plddt": 99},
                           {"sequence": "AAA", "pdbname": "a_seed_2", "iptm": 0.7, "ipsae": 0.9, "plddt": 60}])
    config = SessionConfig(folders=[Folder(str(tmp_path), "T", "af2")], include_phase3=False,
                           phase2=Aggregation(selector="composite_confidence", composite={
                               "weights": {"iptm": 1.0, "ipsae": 3.0},
                               "directions": {"iptm": "higher_better", "ipsae": "higher_better"}, "normalization": "rank"}))
    result = load_predictions(config)
    assert result.table.plddt__af2.iloc[0] == 60
    assert result.table.composite_confidence__af2.iloc[0] == pytest.approx(0.875)
    assert "composite_confidence_std__af2" in result.table
    assert result.table.n_rep_p2.iloc[0] == 2
    assert SessionConfig.from_yaml(config.to_yaml()).to_dict() == config.to_dict()


def test_selector_columns_and_default_composite(tmp_path):
    write_run(tmp_path, 2, [{"sequence": "AAA", "pdbname": "a_seed_1", "iptm": 0.9, "ipsae_binder": 0.1,
                            "apbs_dG_elec": -1, "seed": 1, "pass_filter": True},
                           {"sequence": "AAA", "pdbname": "a_seed_2", "iptm": 0.7, "ipsae_binder": 0.9,
                            "apbs_dG_elec": -10, "seed": 2, "pass_filter": True}])
    write_run(tmp_path, 3, [{"sequence": "AAA", "description": "a_seed_1_0001", "interface_dG": -5, "total_score": -10}])
    folders = [Folder(str(tmp_path), "T", "af2")]
    options, _ = replicate_selector_columns(folders)
    assert {"iptm", "ipsae", "ipsae_binder", "apbs_dG_elec", "replicate_composite"} <= set(options["phase2"])
    assert set(options["phase3"]) == {"interface_dG", "total_score", "replicate_composite"}
    assert not {"seed", "pass_filter", "sequence", "pdbname"}.intersection(options["phase2"])
    config = SessionConfig(folders=folders, phase2=Aggregation(selector="replicate_composite"))
    result = load_predictions(config)
    assert result.table.iptm__af2.iloc[0] == 0.7
    assert result.table.replicate_composite__af2.iloc[0] == pytest.approx(84.375)
    assert result.table.composite__af2.iloc[0] == 100