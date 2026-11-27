import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from valorant_lab.bc_failure_analysis import analyze_failures, trajectory_evidence
from valorant_lab.scenario import contract, load_scenario, validate_scenario

ROOT = Path(__file__).resolve().parents[1]


class FailureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {"goal": [0, 0, 10], "goal_radius": 1, "physics_hz": 60, "action_repeat": 3}
        self.noop = [0, 0, 0, 2, 1, 0, 0, 0]

    def row(self, position, *, action=None, kills=0):
        action = self.noop if action is None else action
        return {"requested_action": action, "info": {"action_applied": action, "position": position,
                "shots": 0, "hits": 0, "kills": kills, "enemies_spawned": 1, "is_success": False,
                "distance_to_goal": 0}}

    def test_idle_is_not_labeled_wall_collision(self):
        result = trajectory_evidence([self.row([0, 0, 0])] * 50, self.config)
        self.assertEqual(result["first_action_index"], None)
        self.assertEqual(result["trailing_noop_steps"], 50)
        self.assertEqual(result["minimum_final_goal_distance_m"], 10)
        self.assertEqual(result["flags"], ["all_noop", "targets_remained_without_fire_request"])

    def test_goal_overshoot_with_enemy_alive_is_not_success(self):
        rows = [self.row([0, 0, 9.5], action=[1, 0, 0, 2, 1, 0, 0, 0]), self.row([0, 0, 12])]
        result = trajectory_evidence(rows, self.config)
        self.assertFalse(result["success"])
        self.assertIn("goal_entered_with_live_targets", result["flags"])
        self.assertIn("left_goal_region", result["flags"])

    def test_turn_in_place_and_shoot_are_active_not_translation(self):
        result = trajectory_evidence([self.row([0, 0, 0], action=[0, 0, 0, 3, 1, 0, 1, 0])], self.config)
        self.assertNotIn("all_noop", result["flags"])
        self.assertIn("no_translation_requested", result["flags"])
        self.assertEqual(result["first_fire_index"], 0)
        self.assertEqual(result["actual_shots"], 0)  # Request is not an actual bullet.

    def test_mismatched_action_and_reset_counter_are_rejected(self):
        row = self.row([0, 0, 0])
        bad = copy.deepcopy(row)
        bad["info"]["action_applied"][0] = 1
        # Deepcopy keeps shared aliases; explicitly separate the request to test a mismatch.
        bad["requested_action"] = self.noop[:]
        with self.assertRaisesRegex(ValueError, "disagree"):
            trajectory_evidence([bad], self.config)
        with self.assertRaisesRegex(ValueError, "cumulative"):
            trajectory_evidence([self.row([0, 0, 0], kills=1), row], self.config)

    def test_test_split_and_training_overlap_cannot_enter_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "valorant_lab.bc_failure_analysis.load_episodes", return_value=[]):
            directory = Path(directory)
            report = directory / "report.json"
            report.write_text(json.dumps({"split": "test", "model_test_only": False}))
            with self.assertRaisesRegex(ValueError, "development"):
                analyze_failures(directory, ROOT / "configs/lotus_shooting.json", [report], directory / "out.json")
            report.write_text(json.dumps({"split": "dev", "model_test_only": False,
                                          "episodes": [{"training_overlap": True}]}))
            with self.assertRaisesRegex(ValueError, "overlaps"):
                analyze_failures(directory, ROOT / "configs/lotus_shooting.json", [report], directory / "out.json")
            self.assertFalse((directory / "out.json").exists())

    def test_collection_variants_keep_original_contract_and_paired_pose(self):
        original = load_scenario(ROOT / "configs/lotus_shooting.json")
        combat = load_scenario(ROOT / "configs/lotus_close_engagement.json")
        clear = load_scenario(ROOT / "configs/lotus_close_clear.json")
        self.assertEqual(contract(original), contract(combat))
        self.assertEqual(contract(original), contract(clear))
        self.assertEqual({k: v for k, v in combat.items() if k not in ("id", "enemy_probability")},
                         {k: v for k, v in clear.items() if k not in ("id", "enemy_probability")})
        self.assertEqual((combat["enemy_probability"], clear["enemy_probability"]), (1, 0))
        self.assertNotEqual(combat["start"], original["start"])

    def test_teacher_hint_does_not_change_contract_or_waive_ordered_gates(self):
        config = load_scenario(ROOT / "configs/lotus_shooting.json")
        hinted = {**config, "teacher_waypoint_start": 3}
        validate_scenario(hinted)
        self.assertEqual(contract(config), contract(hinted))
        for invalid in (-1, len(config["waypoints"]) + 1, True):
            with self.assertRaisesRegex(ValueError, "teacher_waypoint"):
                validate_scenario({**config, "teacher_waypoint_start": invalid})
        with self.assertRaisesRegex(ValueError, "teacher_waypoint"):
            validate_scenario({**hinted, "completion_rule": "ordered_waypoints"})


if __name__ == "__main__":
    unittest.main()
