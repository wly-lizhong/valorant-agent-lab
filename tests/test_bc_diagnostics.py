"""Evidence must distinguish constant outputs from correct observation-conditioned actions."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from valorant_lab.bc_diagnostics import (action_metrics, binary_metrics, comparison_report,
    derangement, stratified_train_indices, validation_separation)

SIZES = [3, 3, 3, 9, 9, 2, 2, 2]


def actions(count):
    return np.tile([1, 0, 0, 4, 4, 0, 0, 0], (count, 1))


def probabilities(predictions):
    result = []
    for head, size in enumerate(SIZES):
        p = np.full((len(predictions), size), .03 / (size - 1))
        p[np.arange(len(predictions)), predictions[:, head]] = .97
        result.append(p)
    return result


def test_constant_majority_policy_fails_rare_actions_despite_high_accuracy():
    target = actions(50)
    target[-1, 6] = 1
    target[-4:, 0] = 0
    target[-6:, 3] = 3
    result = comparison_report(target, probabilities(actions(50)), SIZES, 4)
    m = result["metrics"]
    assert m["heads"]["fire"]["accuracy"] == .98
    assert m["fire"]["recall"] == m["turn"]["recall"] == m["stop"]["recall"] == 0
    assert m["fire"]["false_positive_rate"] == 0
    assert m["constant_action"] and m["unique_joint_actions"] == 1
    d = result["visual_dependency"]
    assert d["shuffled_minus_aligned_nll"] == pytest.approx(0)
    assert d["max_probability_range"] == 0
    assert d["prediction_changed_fraction"] == 0


def test_permuted_inputs_degrade_correct_observation_association():
    target = actions(40)
    target[::2, 6] = 1
    target[::3, 0] = 0
    target[::4, 3] = 2
    result = comparison_report(target, probabilities(target), SIZES, 4)
    assert result["metrics"]["joint_accuracy"] == 1
    assert result["metrics"]["fire"]["recall"] == 1
    d = result["visual_dependency"]
    assert d["fixed_rows"] == 0
    assert d["shuffled_minus_aligned_nll"] > .5
    assert d["aligned_minus_shuffled_joint_accuracy"] > .4
    assert d["max_probability_range"] > .9


def test_no_positives_are_reported_as_undefined_not_perfect_recall():
    result = binary_metrics([False, False], [True, False])
    assert result["recall"] is None and result["precision"] == 0
    assert result["false_positive_rate"] == .5
    assert result["target_positive"] == 0 and result["predicted_positive"] == 1


def test_derangement_has_no_self_rows_and_is_reproducible():
    for n in (2, 3, 50):
        a = derangement(n, 42)
        assert not np.any(a == np.arange(n))
        assert np.array_equal(np.sort(a), np.arange(n))
        assert np.array_equal(a, derangement(n, 42))
    assert derangement(1, 42).tolist() == [0]
    with pytest.raises(ValueError):
        derangement(0, 42)


def test_tiny_selector_retains_rare_behaviors_and_stays_within_train_indices():
    target = actions(500)
    target[-5:, 6] = 1
    target[40:80, 0] = 0
    target[100:160, 3] = 3
    selected = stratified_train_indices(target, 128, 4)
    assert len(selected) == len(np.unique(selected)) == 128
    assert selected.min() >= 0 and selected.max() < len(target)
    assert target[selected, 6].sum() == 5
    assert (target[selected, 0] == 0).sum() > 0
    assert (target[selected, 3] != 4).sum() > 0
    assert np.array_equal(selected, stratified_train_indices(target, 128, 4))
    with pytest.raises(ValueError):
        stratified_train_indices(target, 129, 4)


def test_invalid_categorical_actions_or_probabilities_fail_closed():
    target = actions(2)
    predicted = target.copy()
    predicted[0, 6] = 2
    with pytest.raises(ValueError):
        action_metrics(target, predicted, SIZES, 4)
    p = probabilities(target)
    p[6][:] = np.nan
    with pytest.raises(ValueError):
        comparison_report(target, p, SIZES, 4)


def test_validation_separation_checks_historical_model_training_cases():
    dataset = [(None, {"split": "train", "case_key": "current-train"}),
               (None, {"split": "validation", "case_key": "previous-train"})]
    model_meta = {"episodes": [{"split": "train", "case_key": "current-train"},
                               {"split": "train", "case_key": "previous-train"}]}
    with pytest.raises(ValueError, match="recorded training data"):
        validation_separation(model_meta, dataset)
    clean = {"episodes": [{"split": "train", "case_key": "current-train"}]}
    evidence = validation_separation(clean, dataset)
    assert evidence["verified"] and evidence["overlap_case_keys"] == []
    assert evidence["model_training_case_keys"] == ["current-train"]
    assert evidence["validation_case_keys"] == ["previous-train"]


@pytest.mark.parametrize("model_meta", [{}, {"episodes": []}, {"episodes": [{"split": "train"}]}])
def test_missing_model_training_provenance_cannot_certify_independent_validation(model_meta):
    with pytest.raises(ValueError, match="provenance"):
        validation_separation(model_meta, [(None, {"split": "validation", "case_key": "v"})])


@pytest.fixture
def human_samples(tmp_path):
    from valorant_lab.dataset import EpisodeWriter
    from valorant_lab.env import ROOT
    from valorant_lab.scenario import load_scenario
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    config.update(width=64, height=64, render_width=64, render_height=64)
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(config), encoding="utf-8")

    def add(split):
        writer = EpisodeWriter(tmp_path / "data", config, 42 if split == "train" else 100142,
                               source="script_test", split=split, chunk_size=2)
        target = actions(4)
        target[::2, 6] = 1
        target[::2, 0] = 0
        target[::2, 3] = 3
        for i, action in enumerate(target):
            frame = np.full((64, 64, 3), 40 + i * 50, dtype=np.uint8)
            writer.append(frame, action, before={"position": [999, 999, 999]}, after={}, timing={})
        writer.finish({"is_success": True}, accepted=True)
        return writer.directory
    return config, scenario, add


def tiny_args(tmp_path, scenario):
    return SimpleNamespace(data=str(tmp_path / "data"), scenario=str(scenario), output=str(tmp_path / "tiny"),
        device="cpu", batch_size=2, cpu_threads=1, max_samples=4, steps=2, seed=42,
        policy_variant="normalized", class_balance="sqrt", class_weight_cap=8.,
        learning_rate=3e-4, allow_test_data=True)


def test_tiny_fit_accepts_train_only_and_marks_checkpoint_diagnostic(tmp_path, human_samples):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from valorant_lab.bc_diagnostics import tiny_fit
    _, scenario, add = human_samples
    add("train")
    args = tiny_args(tmp_path, scenario)
    report = tiny_fit(args)
    assert report["validation_used"] is False and report["sample_count"] == 4
    assert report["batch_size"] == 2 and report["requested_batch_size"] == 2
    assert report["device"] == "cpu" and report["precision"] == "fp32"
    assert report["test_only"] and report["diagnostic_only"]
    assert report["after"]["metrics"]["samples"] == 4
    assert all(m["split"] == "train" for m in report["data"]["episodes"])
    meta = json.loads((tmp_path / "tiny/diagnostic-only.json").read_text(encoding="utf-8"))
    assert meta["test_only"] and meta["diagnostic_only"]
    assert meta["kind"] == "tiny_fit_diagnostic"
    assert meta["batch_size"] == 2 and meta["training_device"] == "cpu"
    assert all(m["split"] == "train" for m in meta["episodes"])
    with pytest.raises(FileExistsError):
        tiny_fit(args)


def test_tiny_fit_never_decodes_validation_and_saved_model_diagnostics_keep_splits(tmp_path, human_samples, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from valorant_lab import bc_diagnostics as diagnostic
    _, scenario, add = human_samples
    add("train")
    validation = add("validation")
    original_read = diagnostic.read_demonstration_chunk
    original_dataset = diagnostic.DemonstrationDataset

    def read(path, meta, chunk):
        assert path != validation and meta["split"] == "train"
        return original_read(path, meta, chunk)

    def train_dataset(episodes):
        assert all(meta["split"] == "train" for _, meta in episodes)
        return original_dataset(episodes)

    with monkeypatch.context() as patch:
        patch.setattr(diagnostic, "read_demonstration_chunk", read)
        patch.setattr(diagnostic, "DemonstrationDataset", train_dataset)
        diagnostic.tiny_fit(tiny_args(tmp_path, scenario))
    args = SimpleNamespace(data=str(tmp_path / "data"), scenario=str(scenario),
        model=str(tmp_path / "tiny/diagnostic-only.zip"), output=str(tmp_path / "diagnostic.json"),
        max_samples=0, seed=42, device="cpu", cpu_threads=1, batch_size=2, allow_test_data=True)
    report = diagnostic.diagnose_model(args)
    assert report["model_test_only"]
    assert report["validation_separation"]["verified"]
    assert set(report["splits"]) == {"train", "validation"}
    for split in report["splits"].values():
        assert split["samples"] == split["available_samples"] == 4
        assert split["episode_count"] == 1
        assert split["visual_dependency"]["fixed_rows"] == 0
    assert report["data"]["sha256"] and report["source"]["sha256"] and report["model_sha256"]
    with pytest.raises(FileExistsError):
        diagnostic.diagnose_model(args)
    model_path = tmp_path / "tiny/diagnostic-only.json"
    model_meta = json.loads(model_path.read_text(encoding="utf-8"))
    validation_meta = json.loads((validation / "episode.json").read_text(encoding="utf-8"))
    model_meta["episodes"].append({"split": "train", "case_key": validation_meta["case_key"]})
    model_path.write_text(json.dumps(model_meta), encoding="utf-8")
    args.output = str(tmp_path / "must-not-publish.json")
    with pytest.raises(ValueError, match="recorded training data"):
        diagnostic.diagnose_model(args)
    assert not (tmp_path / "must-not-publish.json").exists()
