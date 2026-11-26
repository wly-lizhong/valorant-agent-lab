import pytest
from valorant_lab.guidance import route_guidance


CONFIG = {"start": [0, 0, 0], "waypoints": [[0, 0, -4], [4, 0, -4]],
          "goal": [4, 0, -4], "goal_radius": 1.1}


@pytest.mark.parametrize("yaw, angle, word", [(0, 0, "按 W"), (-90, 90, "向右"),
                                             (90, -90, "向左"), (180, -180, "身后")])
def test_direction_uses_upstream_yaw_convention(yaw, angle, word):
    guide = route_guidance(CONFIG, {"position": [0, 0, 0], "yaw": yaw})
    assert guide["angle"] == pytest.approx(angle)
    assert guide["distance"] == 4
    assert word in guide["instruction"]


def test_engine_progress_not_client_distance_controls_target():
    guide = route_guidance(CONFIG, {"position": [0, 0, -4], "yaw": 0, "waypoint": 1})
    assert guide["target"] == [4, 0, -4] and guide["completed"] == 1
    assert "向右" in guide["instruction"]
    guide = route_guidance(CONFIG, {"position": [4, 0, -4], "yaw": 0, "waypoint": 2})
    assert not guide["success"] and "交战" in guide["instruction"]
    assert "练习完成" in route_guidance(CONFIG, {
        "position": [4, 0, -4], "yaw": 0, "waypoint": 2, "is_success": True})["instruction"]


def test_missing_telemetry_and_camera_pitch():
    assert route_guidance(CONFIG, {}) is None
    assert "向下" in route_guidance(CONFIG, {"position": [0, 0, 0], "yaw": 0, "view_pitch": 80})["instruction"]
    assert "向上" in route_guidance(CONFIG, {"position": [0, 0, 0], "yaw": 0, "view_pitch": -80})["instruction"]
    assert route_guidance(CONFIG, {"position": [0, 0, 0], "yaw": 0, "view_yaw": 90})["angle"] == -90
