"""Human practice guidance only; never a policy observation or demonstration label."""
import math


def route_guidance(config, info):
    points = config.get("waypoints") or [config["goal"]]
    position = info.get("position")
    yaw = info.get("view_yaw", info.get("yaw"))
    if position is None or yaw is None:
        return None
    completed = min(max(int(info.get("waypoint", 0)), 0), len(points))
    target = points[min(completed, len(points) - 1)]
    dx, dz = target[0] - position[0], target[2] - position[2]
    # Upstream PlayerView: yaw 0 faces -Z, positive yaw turns right (+X).
    angle = (math.degrees(math.atan2(dx, -dz)) - yaw + 180) % 360 - 180
    pitch = info.get("view_pitch", info.get("pitch", 0))
    if info.get("is_success"):
        instruction = "已到终点，练习完成！"
    elif abs(pitch) > 35:
        instruction = "鼠标向下移动，把视线放平" if pitch > 0 else "鼠标向上移动，把视线放平"
    elif abs(angle) > 150:
        instruction = "下一点在身后，先转身再前进"
    elif angle > 15:
        instruction = "鼠标向右转，朝向黄色点后按 W 前进"
    elif angle < -15:
        instruction = "鼠标向左转，朝向黄色点后按 W 前进"
    else:
        instruction = "朝向正确，按 W 沿通道前进"
    if completed == len(points) and not info.get("is_success"):
        instruction = "路线已走完；回到终点圈，并完成本课的交战要求"
    return {"route": [config["start"], *points], "position": position, "yaw": yaw,
            "completed": completed, "total": len(points), "target": target,
            "distance": math.dist(position, target), "angle": angle,
            "radius": config["goal_radius"], "instruction": instruction,
            "success": bool(info.get("is_success"))}
