from io import BytesIO
import zipfile

import pandas as pd
import pytest

from binder_viz.selection import candidate_pool, combine_scores, export_selection, make_aligner, pair_identity, select_candidates


def pool():
    return pd.DataFrame({"sequence_id": ["a", "b", "c", "d"], "sequence": ["AAAAAA", "AAAAAA", "CCCCCC", "DDDDDD"],
                         "target": ["T", "T", "T", "Other"], "iptm__af2": [0.9, 0.8, 0.7, 0.6],
                         "n_rep_p2": [3, 1, 3, 3]})


def test_combined_ranking_rules():
    table = pool().assign(**{"energy__af2": [-1.0, -4.0, -2.0, -3.0]})
    directions = {"iptm__af2": "higher_better", "energy__af2": "lower_better"}
    weights = {"iptm__af2": 1.0, "energy__af2": 1.0}
    mean_rank = combine_scores(table, weights, directions)
    assert mean_rank.iloc[1] > mean_rank.iloc[2]
    assert combine_scores(table, weights, directions, "min_rank").iloc[0] == pytest.approx(1 / 3)
    intersection = combine_scores(table, weights, directions, "intersection", percentile=0.9)
    assert intersection.notna().sum() == 1
    union = combine_scores(table, weights, directions, "union", percentile=0.9)
    assert union.notna().sum() == 3
    chosen = candidate_pool(table.assign(combined=mean_rank), "combined", "higher_better")
    assert chosen[chosen.target.eq("T")].sequence_id.iloc[0] == "b"
    with pytest.raises(ValueError, match="nonzero weight"):
        combine_scores(table, {"iptm__af2": 0.0}, directions)
    with pytest.raises(ValueError, match="Unknown combination rule"):
        combine_scores(table, weights, directions, "median")
    with pytest.raises(ValueError, match="not available"):
        combine_scores(table, {"absent__af2": 1.0}, directions)


def test_alignment_identity_and_selection_limits():
    assert pair_identity(make_aligner(), "AAAA", "AAAA") == 1
    assert pair_identity(make_aligner(), "AAAA", "CCCC") == 0
    candidates = candidate_pool(pool(), "iptm__af2", "higher_better")
    chosen = select_candidates(candidates, 4, per_target=2, score="iptm__af2")
    assert chosen.sequence_id.tolist() == ["a", "c", "d"]
    assert chosen.selection_rationale.str.contains("rank=").all()


def test_target_filter_and_tested_exclusion():
    candidates = candidate_pool(pool(), "iptm__af2", "higher_better", [{"score": "iptm__af2", "min": 0.8, "target": "T"}], {"a"})
    assert candidates.sequence_id.tolist() == ["b", "d"]
    assert candidates.iloc[0]["flags"] == "few phase2 replicates"


def test_agreement_and_exports():
    candidates = candidate_pool(pool(), "iptm__af2", "higher_better", agreement_columns=["iptm__af2"],
                                directions={"iptm__af2": "higher_better"}, minimum_models=1, agreement_percentile=0.9)
    assert candidates.sequence_id.tolist() == ["a", "d"]
    result = export_selection(candidates, {"folders": [], "threshold": 0.9})
    assert "session_config_json" in result["csv"].decode()
    assert ">a target=T" in result["fasta"].decode()
    assert b'"threshold": 0.9' in result["fasta"]
    with zipfile.ZipFile(BytesIO(result["zip"])) as archive:
        assert set(archive.namelist()) == {"selected.csv", "selected.fasta", "session.yaml"}


def specificity_pool():
    return pd.DataFrame({"sequence_id": ["a1", "a2", "a3", "b1", "b2", "b3"],
                         "sequence": ["AAAA", "CCCC", "DDDD", "AAAA", "CCCC", "CCCC"],
                         "target": ["A", "A", "A", "B", "B", "B"],
                         "iptm__af2": [0.95, 0.9, 0.8, 0.85, 0.2, 0.3],
                         "ros_dg__af2": [-10, -9, -8, -8, -2, -3]})


@pytest.mark.parametrize("score,direction,threshold", [("iptm__af2", "higher_better", 0.7),
                                                       ("ros_dg__af2", "lower_better", -5)])
def test_negative_selection_matches_sequence_across_targets(score, direction, threshold):
    rule = {"target": "B", "score": score, "direction": direction, "threshold": threshold}
    result = candidate_pool(specificity_pool(), "iptm__af2", "higher_better", positive_targets=["A"],
                            negative_selection=[rule])
    assert result.sequence_id.tolist() == ["a2"]
    assert result.negative_targets_checked.tolist() == ["B"]
    assert result[f"off_target__B__{score}"].iloc[0] == (0.3 if direction == "higher_better" else -3)


def test_negative_selection_missing_evidence_and_experiments():
    rule = {"target": "B", "score": "iptm__af2", "direction": "higher_better", "threshold": 0.7}
    result = candidate_pool(specificity_pool(), "iptm__af2", "higher_better", negative_selection=[rule],
                            exclude_missing_off_targets=False)
    assert result.sequence_id.tolist() == ["a2", "a3"]
    assert result.iloc[1]["off_target__B__status"] == "missing evidence"
    experiments = pd.DataFrame({"target": ["B", "B", "B"], "sequence": ["CCCC", "DDDD", "DDDD"],
                                "binds": [True, False, None]})
    result = candidate_pool(specificity_pool(), "iptm__af2", "higher_better", negative_selection=[rule],
                            off_target_experiments=experiments)
    assert result.sequence_id.tolist() == ["a3"]


def test_experimental_only_negative_selection_and_overlap():
    experiments = pd.DataFrame({"target": ["B", "B"], "sequence": ["AAAA", "CCCC"], "binds": [True, False]})
    result = candidate_pool(specificity_pool(), "iptm__af2", "higher_better", negative_selection=[{"target": "B"}],
                            off_target_experiments=experiments)
    assert result.sequence_id.tolist() == ["a2"]
    with pytest.raises(ValueError, match="overlap"):
        candidate_pool(specificity_pool(), "iptm__af2", "higher_better", positive_targets=["B"],
                       negative_selection=[{"target": "B"}])


def test_off_target_evidence_uses_unfiltered_predictions():
    table = specificity_pool()
    result = candidate_pool(table[table.target.eq("A")], "iptm__af2", "higher_better",
                            negative_selection=[{"target": "B", "score": "iptm__af2", "direction": "higher_better", "threshold": 0.7}],
                            off_target_predictions=table)
    assert result.sequence_id.tolist() == ["a2"]


def test_missing_nullable_experimental_labels_do_not_override_missing_policy():
    measurements = pd.DataFrame({"target": ["B", "B"], "sequence": ["AAAA", "CCCC"],
                                 "binds": pd.array([True, pd.NA], dtype="boolean")})
    result = candidate_pool(specificity_pool(), "iptm__af2", "higher_better", negative_selection=[{"target": "B"}],
                            off_target_experiments=measurements, exclude_missing_off_targets=False)
    assert result.sequence_id.tolist() == ["a2", "a3"]
    assert result["flags"].str.contains("missing off-target evidence").all()