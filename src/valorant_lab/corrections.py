"""Development failures become restartable tasks, never arbitrary-state snapshots."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import write_json
from .scenario import digest

LABELS = {"look_limit": "调整视角后继续原路线", "blocked_movement": "脱离障碍并继续前进",
          "reload_needed": "安全换弹并完成原任务", "combat": "处理可见敌人后继续路线",
          "route_incomplete": "完成路线并处理沿途可见敌人", "fall": "沿可通行地面完成路线"}


def classify_failure(trace: list[dict]) -> str:
    if not trace:
        return "route_incomplete"
    tail = trace[-40:]
    last = trace[-1]
    if last.get("termination_reason") == "fall":
        return "fall"
    if len(tail) >= 20 and np.mean([abs(x.get("pitch", 0)) > 70 for x in tail]) > .8:
        return "look_limit"
    if last.get("ammo") == 0 and not last.get("reloading"):
        return "reload_needed"
    if last.get("termination_reason") == "death":
        return "combat"
    if len(tail) >= 20:
        positions = np.array([x.get("position", [0, 0, 0]) for x in tail])
        moving = [any(x.get("action_applied", [0, 0])[:2]) for x in tail]
        if np.mean(moving) > .7 and np.max(np.linalg.norm(positions - positions[0], axis=1)) < .5:
            return "blocked_movement"
    return "route_incomplete"


def enqueue_failure(queue, config, engine_seed, trace, image, *, split, model_hash):
    if split != "dev":
        raise ValueError("Only development evaluations can create correction tasks; keep test/validation held out")
    reason = classify_failure(trace)
    last = trace[-1]
    cell = [round(float(v) / 2) for v in last.get("position", [0, 0, 0])]
    key = digest([config.get("family", config["id"]), reason, last.get("waypoint", 0), cell])[:16]
    directory = Path(queue) / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "case.json"
    if path.exists():
        case = json.loads(path.read_text(encoding="utf-8"))
        case["occurrences"] += 1
        case["last_model_hash"] = model_hash
        if case.get("status") == "model_passed_once":
            case["status"] = "pending"
        write_json(path, case)
        return path
    case = {"schema": "valorant-correction-v1", "id": key, "split": split, "status": "pending",
            "reason": reason, "title": LABELS[reason], "task": config.get("task", "Complete the route"),
            "scenario": config, "engine_seed": engine_seed, "occurrences": 1,
            "model_hash": model_hash, "last_model_hash": model_hash,
            "replay": "restart_from_seed", "note": "从本次回合起点重做；当前不支持恢复任意时刻快照。"}
    write_json(path, case)
    write_json(directory / "failure-context.json", {"trace": trace[-80:]})
    Image.fromarray(image).save(directory / "failure.png")
    return path


def load_case(path):
    path = Path(path)
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("schema") != "valorant-correction-v1" or case.get("split") != "dev":
        raise ValueError("Invalid correction case or protected test split")
    return case


def update_case(path, **changes):
    case = load_case(path)
    case.update(changes)
    write_json(Path(path), case)


def list_queue(args):
    cases = [(p, load_case(p)) for p in Path(args.queue).glob("*/case.json")]
    cases.sort(key=lambda item: -item[1]["occurrences"])
    for path, c in cases:
        print(f"{c['status']:18} {c['occurrences']:5} 次  {c['title']}  {path}")
    if not cases:
        print("暂无纠错题。先运行 evaluate --split dev --queue <目录>。")
