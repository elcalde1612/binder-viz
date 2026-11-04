import numpy as np
import pandas as pd
import pytest

from binder_viz.analysis import add_consolidation_scores, composite, consensus, consolidation_scores, evaluate, fit_composite, paired_auroc
from binder_viz.experiments import attach_experiments, load_experiments, parse_kd, sequence_labels
from binder_viz.loader import sequence_id
from binder_viz.registry import infer_direction, score_catalog


def cohort():
    return pd.DataFrame({"sequence_id": [str(value) for value in range(12)], "target": ["A"] * 6 + ["B"] * 6,
                         "binds": [0, 0, 0, 1, 1, 1] * 2, "binding_ord": [0, 0, 0, 1, 2, 3] * 2,
                         "good__af2": list(range(6)) * 2, "good__opendde": list(range(6)) * 2,
                         "bad__af2": list(range(5, -1, -1)) * 2, "source": ["internal"] * 12,
                         "log10_kd": list(range(6, 0, -1)) * 2})


def test_sequence_bootstrap_not_replicates():
    data = cohort()
    original = evaluate(data, "good__af2", n_boot=50)
    repeated = evaluate(pd.concat([data, data.iloc[[0, 0, 1]]]), "good__af2", n_boot=50)
    assert original == repeated
    assert original["value"] == 1
    assert original["ci_low"] == pytest.approx(1)
    assert original["n"] == 12
    assert original["positives"] == 6
    assert evaluate(data, "good__af2", "log10_kd", "Spearman")["value"] == 1


def test_sparse_and_constant_scores():
    data = cohort().iloc[:3]
    assert np.isnan(evaluate(data, "good__af2")["value"])
    data = cohort().assign(constant=1)
    assert np.isnan(evaluate(data, "constant")["value"])


def test_direction_catalog_confirmation():
    data = cohort()
    assert infer_direction(data["good__af2"], data.binds) == "higher_better"
    assert infer_direction(data["bad__af2"], data.binds) == "lower_better"
    catalog, _ = score_catalog(data, [], data)
    assert not catalog.confirmed.any()
    assert catalog.set_index("column").loc["good__af2", "direction"] == "higher_better"


def test_composites_consensus_and_paired_bootstrap():
    data = cohort()
    directions = {"good__af2": "higher_better", "good__opendde": "higher_better", "bad__af2": "lower_better"}
    result = composite(data, {"good__af2": 1, "bad__af2": 1}, directions)
    assert result.iloc[0] < result.iloc[5]
    result = consensus(data, ["good__af2", "good__opendde"], directions, "min")
    assert result.iloc[5] == 1
    delta = paired_auroc(data, "good__af2", "good__opendde", directions, n_boot=40)
    assert delta["delta_auroc"] == 0
    assert delta["bootstrap_p"] == 1
    with pytest.raises(ValueError, match="one score per model"):
        consensus(data, ["good__af2", "bad__af2"], directions)


def test_target_grouped_fit():
    data = cohort()
    result, coefficients = fit_composite(data, ["good__af2", "bad__af2"])
    assert result.notna().all()
    assert set(coefficients.held_out_targets) == {"A", "B"}
    assert result.iloc[0] < result.iloc[5]
    with pytest.raises(ValueError, match="two labeled targets"):
        fit_composite(data.iloc[:6], ["good__af2"])


def test_default_composite_matches_consolidation_group_weights():
    frame = pd.DataFrame({"target": ["T", "T"], "iptm": [0.8, 0.2], "ipsae": [0.1, 0.9],
                          "apbs_dG_elec": [-10, -1], "apbs_elec_complementarity": [0.2, 0.8],
                          "prodigy_dG_bind": [-5, -8], "prodigy_log10_kd": [-9, -7],
                          "interface_dG": [-10, -2], "interface_sc": [0.9, 0.1],
                          "interface_packstat": [0.8, 0.4], "dG_cross/dSASAx100": [-2, -1]})
    result = consolidation_scores(frame)
    assert result.score_refold.tolist() == [75, 75]
    assert result.score_electrostatics.tolist() == [75, 75]
    assert result.score_rosetta.tolist() == [100, 50]
    assert result.composite.tolist() == pytest.approx([80, 70])
    assert result.pct_interface_dG.tolist() == [100, 50]


def test_default_composite_ties_missing_groups_and_target_scopes():
    frame = pd.DataFrame({"target": ["A", "A", "A", "B"], "iptm": [1, 1, 3, 10]})
    assert consolidation_scores(frame).composite.tolist() == [50, 50, 100, 100]
    assert consolidation_scores(frame, within_target=False).composite.tolist() == [37.5, 37.5, 75, 100]
    assert consolidation_scores(pd.DataFrame({"target": ["A"], "unrelated": [1]})).composite.isna().all()
    frame = pd.DataFrame({"target": ["A", "A"], "iptm": [0.2, 0.9], "interface_dG": [None, -10]})
    assert pd.isna(consolidation_scores(frame).composite.iloc[0])


def test_default_composite_averages_shared_phases_and_sequences():
    table = pd.DataFrame({"sequence_id": ["a", "a", "b"], "target": ["T"] * 3,
                          "iptm__af2": [0.9, 0.9, 0.1], "apbs_dG_elec__af2": [-10, -10, -2],
                          "ros_apbs_dG_elec__af2": [10, 10, -2], "ros_interface_dG__af2": [-8, -8, -2]})
    result = add_consolidation_scores(table, ["af2"])
    assert result.pct_apbs_dG_elec__af2.tolist() == [50, 50, 100]
    assert result.composite__af2.tolist() == pytest.approx([85, 85, 65])
    catalog, _ = score_catalog(result)
    metadata = catalog.set_index("column").loc["composite__af2"]
    assert metadata.stage == "derived"
    assert metadata.direction == "higher_better"
    assert metadata.confirmed


@pytest.mark.parametrize("value,expected,qualifier", [(">10 uM", 10000, ">"), ("0.5 nM", 0.5, "="), ("<2e-8 M", 20, "<"), ("no binding detected", 20000, ">")])
def test_kd_parser(value, expected, qualifier):
    parsed, actual_qualifier = parse_kd(value, 20000)
    assert parsed == pytest.approx(expected)
    assert actual_qualifier == qualifier


def test_experimental_mapping_sources_duplicates(tmp_path):
    path = tmp_path / "internal.csv"
    pd.DataFrame({"seq": ["AAA", "CCC", "DDD", "AAA"], "label": ["no", "strong", "Nan", "weak"], "kd": [">10 uM", "5", "no binding detected", "7"]}).to_csv(path, index=False)
    public = tmp_path / "public.csv"
    pd.DataFrame({"sequence": ["AAA"], "binding_class": ["strong"], "kd_nM": [1]}).to_csv(public, index=False)
    predictions = pd.DataFrame({"sequence_id": [sequence_id("T", "AAA"), sequence_id("T", "CCC")], "target": ["T", "T"]})
    result = load_experiments([{"path": str(path), "columns": {"sequence": "seq", "binding_class": "label", "kd_nM": "kd"}},
                               {"path": str(public), "source": "public"}], predictions)
    assert len(result.unmatched) == 1
    assert len(result.duplicates) == 2
    assert result.table[result.table.source.eq("public")].log10_kd.isna().all()
    assert result.table.iloc[0].binds == False
    assert result.table.iloc[1].binds == True
    assert result.table.iloc[2].binding_ord == 0


def test_conflicting_labels_not_pooled_as_independent():
    data = cohort()
    conflict = data.iloc[[0]].assign(binds=1)
    assert evaluate(pd.concat([data, conflict]), "good__af2")["n"] == 11


def test_experimental_identifiers_preserved_without_changing_sequence_matching(tmp_path):
    path = tmp_path / "named.csv"
    path.write_text("target,sequence,sequence_id,id,name,binding\nT,AAAA,external-a,001,Named A,True\nT,CCCC,external-c,002,,False\n")
    predictions = pd.DataFrame({"target": ["T", "T"], "sequence": ["AAAA", "CCCC"],
                                "sequence_id": [sequence_id("T", "AAAA"), sequence_id("T", "CCCC")]})
    result = load_experiments([{"path": str(path)}], predictions)
    assert not result.warnings
    assert result.unmatched.empty
    assert result.table.experimental_id.tolist() == ["001", "002"]
    assert result.table.sequence_id.tolist() == predictions.sequence_id.tolist()
    attached = attach_experiments(predictions, result.table)
    assert sequence_labels(attached).tolist() == ["Named A", "002"]
    assert attached.binds.tolist() == [True, False]
    path.write_text("target,sequence,sequence_id,binding\nT,AAAA,external-a,True\n")
    result = load_experiments([{"path": str(path)}], predictions)
    assert result.table.experimental_id.tolist() == ["external-a"]
    assert result.unmatched.empty