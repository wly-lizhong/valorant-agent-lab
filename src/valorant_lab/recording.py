"""Local first-person demonstration collector. No access to OS-global keys or the official game."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .corrections import load_case, update_case
from .dataset import EpisodeWriter, split_seed, write_json
from .env import ValorantEnv
from .scenario import LEGACY_YAW, LEGACY_PITCH


def look_action(config, dx, dy, degrees_per_pixel):
    yaw = np.asarray(config.get("yaw_bins", LEGACY_YAW))
    pitch = np.asarray(config.get("pitch_bins", LEGACY_PITCH))
    return int(np.argmin(abs(yaw - dx * degrees_per_pixel))), int(np.argmin(abs(pitch + dy * degrees_per_pixel)))


def record(args):
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Install the optional collector: pip install -e '.[collect]'") from exc
    case = load_case(args.case) if args.case else None
    if case and args.split != "train":
        raise ValueError("Correction demonstrations are training data, never validation data")
    scenario = args.scenario
    if case:
        scenario = Path(args.case).parent / "record-scenario.json"
        write_json(scenario, case["scenario"])
    writer = None
    pygame.init()
    try:
        with ValorantEnv(scenario) as env:
            view_width, view_height = 960, 540
            screen = pygame.display.set_mode((view_width, view_height + 100))
            pygame.display.set_caption("Agent Lab 人工示范")
            font = pygame.font.SysFont("Microsoft YaHei,SimHei,Noto Sans CJK SC,DejaVu Sans", 20)
            clock = pygame.time.Clock()
            saved = attempts = 0
            active = False
            jump_ready = False
            state = "ready"
            engine_seed = case["engine_seed"] if case else split_seed(args.split, args.seed)
            frame, info = env.reset(options={"engine_seed": engine_seed})
            observed_at = time.monotonic()

            def capture(enabled):
                pygame.event.set_grab(enabled)
                pygame.mouse.set_visible(not enabled)
                pygame.mouse.get_rel()

            while saved < args.episodes:
                begin = time.monotonic()
                pressed = []
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        pressed.append(event.key)
                    if event.type == pygame.WINDOWFOCUSLOST and active:
                        writer.issue("window_lost_focus")
                        writer.finish(info, accepted=False, complete=False)
                        writer = None
                        active = False; state = "review"; capture(False)
                if pygame.K_q in pressed and not active:
                    return
                if pygame.K_ESCAPE in pressed and active:
                    writer.finish(info, accepted=False, complete=False)
                    writer = None
                    active = False; state = "review"; capture(False)
                if state == "ready" and pygame.K_SPACE in pressed:
                    writer = EpisodeWriter(args.output, env.config, engine_seed, split=args.split,
                                           case_id=case["id"] if case else None)
                    active = True; state = "recording"; capture(True)
                    jump_ready = False
                elif state == "review":
                    if pygame.K_s in pressed and writer is not None:
                        directory = writer.finish(info, accepted=bool(info.get("is_success")))
                        if writer.manifest["accepted"]:
                            saved += 1
                            if case:
                                update_case(args.case, status="demonstrated", demonstration=directory.name)
                        print(f"示范保存：{directory}；可用于训练：{writer.manifest['accepted']}", flush=True)
                        writer = None
                        state = "next"
                    elif pygame.K_r in pressed:
                        if writer is not None:
                            writer.finish(info, accepted=False)
                            writer = None
                        state = "next"
                if state == "next" and saved < args.episodes:
                    attempts += 1
                    engine_seed = case["engine_seed"] if case else split_seed(args.split, args.seed + attempts)
                    frame, info = env.reset(options={"engine_seed": engine_seed})
                    observed_at = time.monotonic()
                    state = "ready"
                if active:
                    keys = pygame.key.get_pressed()
                    action = env.noop.copy()
                    action[0] = 1 if keys[pygame.K_w] and not keys[pygame.K_s] else 2 if keys[pygame.K_s] and not keys[pygame.K_w] else 0
                    action[1] = 1 if keys[pygame.K_a] and not keys[pygame.K_d] else 2 if keys[pygame.K_d] and not keys[pygame.K_a] else 0
                    action[2] = 2 if keys[pygame.K_LCTRL] else 1 if keys[pygame.K_LSHIFT] else 0
                    dx, dy = pygame.mouse.get_rel()
                    action[3], action[4] = look_action(env.config, dx, dy, args.mouse_scale)
                    if not keys[pygame.K_SPACE]:
                        jump_ready = True
                    action[5] = int(jump_ready and keys[pygame.K_SPACE])
                    action[6] = int(pygame.mouse.get_pressed()[0])
                    action[7] = int(pygame.K_r in pressed)
                    input_at = time.monotonic()
                    next_frame, _, terminated, truncated, after = env.step(action)
                    returned_at = time.monotonic()
                    writer.append(frame, action, before=info, after=after,
                                  timing={"observation_received": observed_at, "input_sampled": input_at,
                                          "response_received": returned_at, "clock": "paced_lockstep",
                                          "mouse_dx": dx, "mouse_dy": dy, "degrees_per_pixel": args.mouse_scale})
                    frame, info, observed_at = next_frame, after, returned_at
                    if terminated or truncated:
                        active = False; state = "review"; capture(False)
                surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                screen.fill((22, 25, 30))
                screen.blit(pygame.transform.scale(surface, (view_width, view_height)), (0, 0))
                if active:
                    line = "录制中：WASD 移动 / 鼠标瞄准 / 左键开枪 / R 换弹 / Shift 静步 / Ctrl 蹲 / Esc 放弃"
                elif state == "review":
                    line = ("完成！S 保存示范 / R 重试 / Q 退出" if info.get("is_success")
                            else "未完成：R 重试 / Q 退出；失败记录不会成为正确示范")
                else:
                    line = "Space 开始 / Q 退出；沿 A Root 前往入口，处理可见敌人后继续路线"
                screen.blit(font.render(line, True, (235, 235, 235)), (12, view_height + 12))
                if case:
                    screen.blit(font.render("本题：" + case["title"] + "（从回合起点重做）", True, (130, 205, 230)),
                                (12, view_height + 45))
                else:
                    screen.blit(font.render(f"数据分组：{args.split}   已保存：{saved}/{args.episodes}", True, (130, 205, 230)),
                                (12, view_height + 45))
                pygame.display.flip()
                if active:
                    time.sleep(max(0, env.config["action_repeat"] / env.config["physics_hz"] - (time.monotonic() - begin)))
                else:
                    clock.tick(30)
    finally:
        if writer is not None and not writer.closed:
            writer.finish({}, accepted=False, complete=False)
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        pygame.quit()
