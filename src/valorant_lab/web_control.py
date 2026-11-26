"""Browser demonstrations: lossless observed frames, explicit presentation ACKs, one owner.

The engine is stepped only after input for its displayed frame. This avoids silently
pairing delayed browser actions with newer unseen observations. It is not realtime streaming.
"""
from __future__ import annotations
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit
import uuid
from PIL import Image
from .corrections import load_case, update_case
from .dataset import EpisodeWriter, split_seed, write_json
from .env import ROOT, ValorantEnv
from .recording import look_action
from .scenario import load_scenario, validate_scenario

COURSES = {
    "lotus_route": ("01 · 路线行走", "沿 A Root 向 A 入口前进，转弯后继续走到终点。"),
    "lotus_shooting": ("02 · 停稳射击", "沿路线前进，停稳后击杀可见敌人，再继续到入口。"),
    "lotus_reload": ("03 · 换弹与前进", "初始弹匣为空，换弹后处理可见敌人并完成路线。"),
    "lotus_recovery": ("04 · 调整视角", "先把视角恢复到正常高度，再完成同一条路线。"),
    "lotus_movement_combat": ("05 · 组合交战", "沿路线前进，处理可能出现的回击敌人，必要时换弹。"),
}
KEYS = {"KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft", "ControlLeft", "Space"}


def number(value, name, lo=0, hi=1e12):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"Invalid {name}")
    return value


@dataclass
class Settings:
    root: Path = ROOT
    output: Path = ROOT / "runs/demonstrations"
    queue: Path = ROOT / "runs/ppo-movement-combat/corrections"
    state_dir: Path = ROOT / "runs/web"
    max_frame_age_ms: float = 250
    control_timeout: float = 3
    idle_timeout: float = 120
    source: str = "human"  # Non-human test harnesses must override; no production CLI switch.


class Catalog:
    def __init__(self, settings):
        self.settings = settings

    def items(self):
        items = [{"id": key, "title": title, "description": description, "kind": "course"}
                 for key, (title, description) in COURSES.items()]
        directory = self.settings.queue.resolve()
        cases = []
        for path in directory.glob("*/case.json"):
            if not re.fullmatch(r"[0-9a-f]{16}", path.parent.name) or path.resolve().parent.parent != directory:
                continue
            try:
                case = load_case(path)
                cases.append({"id": "case:" + path.parent.name, "title": case["title"],
                              "description": "从回合起点重做；成功后提交示范。", "kind": "correction",
                              "occurrences": case["occurrences"], "status": case["status"]})
            except (ValueError, KeyError, OSError):
                continue
        return items + sorted(cases, key=lambda c: -c["occurrences"])

    def resolve(self, task):
        if isinstance(task, str) and task in COURSES:
            return load_scenario(self.settings.root / "configs" / (task + ".json")), None, None
        if isinstance(task, str) and re.fullmatch(r"case:[0-9a-f]{16}", task):
            directory = self.settings.queue.resolve()
            path = directory / task[5:] / "case.json"
            if path.resolve().parent.parent != directory:
                raise ValueError("Case escaped the configured queue")
            case = load_case(path)
            validate_scenario(case["scenario"])
            if case["scenario"].get("mode") != "move_combat":
                raise ValueError("Browser corrections require movement/combat scenarios")
            return case["scenario"], path, case
        raise ValueError("Unknown course/correction task")


class HumanSession:
    """All methods run on one worker; the engine and writer are never concurrently accessed."""
    def __init__(self, settings, env_factory=ValorantEnv):
        self.settings, self.env_factory = settings, env_factory
        self.env = self.writer = None
        self.state = "idle"
        self.frame_id = 0
        self.info = {}
        self.presented = None
        self.received_at = self.sent_at = 0.

    def prepare(self, request):
        if self.state not in ("idle", "saved"):
            raise ValueError("Finish or discard this episode before opening another")
        self.config, self.case_path, self.case = Catalog(self.settings).resolve(request.get("task"))
        self.split = request.get("split", "train")
        seed_index = request.get("seed", 0)
        if type(seed_index) is not int or self.split not in ("train", "validation"):
            raise ValueError("Invalid demonstration split/seed")
        self.seed = split_seed(self.split, seed_index)
        if self.case:
            if self.split != "train":
                raise ValueError("Correction cases are training data, never validation")
            self.seed = self.case["engine_seed"]
        self.mouse_scale = number(request.get("mouse_scale", .08), "mouse scale", .001, 2)
        self.writer = None
        directory = self.settings.state_dir / ("session-" + uuid.uuid4().hex[:16])
        directory.mkdir(parents=True)
        self.scenario_path = directory / "scenario.json"
        write_json(self.scenario_path, self.config)
        self.env = self.env_factory(self.scenario_path)
        self.frame, self.info = self.env.reset(options={"engine_seed": self.seed})
        self.received_at = time.monotonic()
        self.state = "ready"
        return self.packet()

    def packet(self):
        self.frame_id += 1
        self.presented = None
        buffer = io.BytesIO()
        Image.fromarray(self.frame).save(buffer, format="PNG", compress_level=1)
        self.sent_at = time.monotonic()
        return {"type": "frame", "state": self.state, "frame_id": self.frame_id,
                "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "width": self.config["width"], "height": self.config["height"],
                "period_ms": self.config["action_repeat"] / self.config["physics_hz"] * 1000,
                "success": bool(self.info.get("is_success")),
                "reason": self.info.get("termination_reason", ""),
                "quality_ok": not self.writer or not self.writer.manifest["quality_issues"],
                "step": self.info.get("step", 0), "source": self.settings.source}

    def start(self):
        if self.state != "ready":
            raise ValueError("Open a course before starting")
        self.writer = EpisodeWriter(self.settings.output, self.config, self.seed, split=self.split,
                                   case_id=self.case["id"] if self.case else None, source=self.settings.source)
        self.writer.manifest["collection"] = {"transport": "browser_png_lockstep_v1",
            "max_frame_age_ms": self.settings.max_frame_age_ms, "mouse_scale": self.mouse_scale,
            "engine": getattr(self.env, "engine_metadata", {})}
        self.state = "active"
        return self.packet()

    def acknowledge(self, request, received_at):
        if self.state != "active" or type(request.get("frame_id")) is not int or request["frame_id"] != self.frame_id:
            raise ValueError("Displayed frame does not match the current observation")
        if self.presented is not None:
            raise ValueError("Duplicate frame presentation")
        self.presented = {"browser_ms": number(request.get("displayed_ms"), "display timestamp"),
                          "server_at": received_at}

    def step(self, request, received_at):
        if (self.state != "active" or not self.presented or type(request.get("frame_id")) is not int
                or request["frame_id"] != self.frame_id):
            raise ValueError("Input must reference the current acknowledged frame")
        sampled = number(request.get("sampled_ms"), "input timestamp")
        if not 0 <= sampled - self.presented["browser_ms"] <= 10000:
            raise ValueError("Input timestamp precedes the displayed frame or is stale")
        keys = request.get("keys", [])
        if not isinstance(keys, list) or len(keys) > len(KEYS) or any(not isinstance(k, str) or k not in KEYS for k in keys):
            raise ValueError("Invalid keys")
        if any(type(request.get(k)) is not bool for k in ("fire", "reload", "focused")) or not request["focused"]:
            raise ValueError("Input requires focused browser control")
        dx = number(request.get("dx", 0), "mouse dx", -10000, 10000)
        dy = number(request.get("dy", 0), "mouse dy", -10000, 10000)
        action = self.env.noop.copy()
        action[0] = 1 if "KeyW" in keys and "KeyS" not in keys else 2 if "KeyS" in keys and "KeyW" not in keys else 0
        action[1] = 1 if "KeyA" in keys and "KeyD" not in keys else 2 if "KeyD" in keys and "KeyA" not in keys else 0
        action[2] = 2 if "ControlLeft" in keys else 1 if "ShiftLeft" in keys else 0
        action[3], action[4] = look_action(self.config, dx, dy, self.mouse_scale)
        action[5], action[6], action[7] = int("Space" in keys), int(request["fire"]), int(request["reload"])
        age = (received_at - self.sent_at) * 1000
        if age > self.settings.max_frame_age_ms:
            self.writer.issue("browser_frame_age_exceeded")
        timing = {"clock": "browser_paced_lockstep", "browser_frame_id": self.frame_id,
                  "browser_displayed_ms": self.presented["browser_ms"], "browser_input_ms": sampled,
                  "server_frame_received": self.received_at, "server_frame_sent": self.sent_at,
                  "server_display_ack": self.presented["server_at"], "server_input_received": received_at,
                  "frame_age_ms": age, "mouse_dx": dx, "mouse_dy": dy, "degrees_per_pixel": self.mouse_scale}
        self.presented = None  # consume this frame exactly once, before advancing the world
        timing["server_step_started"] = time.monotonic()
        after_frame, _, done, truncated, after = self.env.step(action)
        timing["server_step_finished"] = time.monotonic()
        self.writer.append(self.frame, action, before=self.info, after=after, timing=timing)
        self.frame, self.info = after_frame, after
        self.received_at = timing["server_step_finished"]
        if done or truncated:
            self.state = "review"
            self.close_engine()
        packet = self.packet()
        packet["frame_age_ms"] = age
        packet["engine_step_ms"] = (timing["server_step_finished"] - timing["server_step_started"]) * 1000
        return packet

    def save(self):
        if self.state != "review" or self.writer is None:
            raise ValueError("Only a completed episode can be submitted")
        directory = self.writer.finish(self.info, accepted=bool(self.info.get("is_success")))
        accepted = self.writer.manifest["accepted"]
        if accepted and self.case:
            update_case(self.case_path, status="demonstrated", demonstration=directory.name)
        self.state = "saved"
        return {"type": "saved", "accepted": accepted, "episode": directory.name,
                "quality_issues": self.writer.manifest["quality_issues"]}

    def close_engine(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def abort(self, reason="discarded"):
        try:
            if self.writer is not None and not self.writer.closed:
                self.writer.issue(reason)
                self.writer.finish(self.info, accepted=False, complete=False)
        finally:
            self.close_engine()
            self.writer = None
            self.state = "idle"
        return {"type": "stopped", "reason": reason}


def token_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Token file must not be a symlink")
    if not path.exists():
        with path.open("x", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            handle.write(secrets.token_urlsafe(32))
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise ValueError("Token file permissions must be 600")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError("Use a token of at least 32 characters")
    return token


async def shutdown_controller(controller, executor):
    # Peer disconnect can cancel the request while abort is still queued. Shield
    # the submitted job so both the dataset and engine are closed before releasing
    # the control lease. Keeping only executor.shutdown() does not prevent a queued
    # cleanup future from being cancelled before its worker starts.
    cleanup = asyncio.get_running_loop().run_in_executor(
        executor, partial(controller.abort, "connection_closed"))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await asyncio.shield(cleanup)
        raise
    finally:
        executor.shutdown(wait=True)


def create_app(settings, *, token, origins, env_factory=ValorantEnv):
    from aiohttp import web, WSMsgType
    origins = frozenset(origins)
    hosts = {urlsplit(origin).netloc for origin in origins}
    sessions = {}
    holder = {"ws": None}

    @web.middleware
    async def guard(request, handler):
        if request.host not in hosts:
            raise web.HTTPForbidden(text="Unrecognized host")
        if request.method == "POST" or request.path == "/ws":
            if request.headers.get("Origin") not in origins:
                raise web.HTTPForbidden(text="Unrecognized origin")
        if request.path.startswith("/api/") and request.path != "/api/login" or request.path == "/ws":
            if sessions.get(request.cookies.get("agent_session"), 0) <= time.monotonic():
                raise web.HTTPUnauthorized(text="请先登录")
        response = await handler(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"})
        return response

    app = web.Application(middlewares=[guard], client_max_size=16384)

    async def login(request):
        try:
            data = await request.json()
        except (ValueError, TypeError):
            raise web.HTTPBadRequest(text="Invalid login")
        if (not isinstance(data, dict) or not isinstance(data.get("token"), str)
                or not secrets.compare_digest(data["token"].encode("utf-8"), token.encode("utf-8"))):
            await asyncio.sleep(.25)
            raise web.HTTPUnauthorized(text="访问口令不正确")
        expired = [key for key, until in sessions.items() if until <= time.monotonic()]
        for key in expired:
            del sessions[key]
        if len(sessions) >= 128:
            raise web.HTTPTooManyRequests(text="Too many sessions; restart or wait for expiry")
        session = secrets.token_urlsafe(32)
        sessions[session] = time.monotonic() + 12 * 3600
        response = web.json_response({"ok": True})
        response.set_cookie("agent_session", session, httponly=True, samesite="Strict",
                            secure=request.headers["Origin"].startswith("https://"), max_age=12 * 3600)
        return response

    async def catalog(_request):
        return web.json_response({"tasks": Catalog(settings).items(), "source": settings.source,
                                  "busy": holder["ws"] is not None, "max_frame_age_ms": settings.max_frame_age_ms})

    async def websocket(request):
        if holder["ws"] is not None:
            raise web.HTTPConflict(text="已有一个人工会话在控制游戏，请先结束它")
        ws = web.WebSocketResponse(max_msg_size=16384, heartbeat=10, compress=False)
        holder["ws"] = ws  # reserve before the first await
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-human")
        controller = HumanSession(settings, env_factory)
        loop = asyncio.get_running_loop()
        async def work(fn, *args):
            return await loop.run_in_executor(executor, partial(fn, *args))
        try:
            await ws.prepare(request)
            await ws.send_json({"type": "connected"})
            while not ws.closed:
                timeout = settings.control_timeout if controller.state == "active" else settings.idle_timeout
                message = await ws.receive(timeout=timeout)
                received = time.monotonic()
                if sessions.get(request.cookies.get("agent_session"), 0) <= received:
                    raise ValueError("登录已过期，请重新登录")
                if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
                if message.type != WSMsgType.TEXT:
                    raise ValueError("Only JSON controls are accepted")
                data = json.loads(message.data)
                if not isinstance(data, dict):
                    raise ValueError("Invalid control message")
                kind = data.get("type")
                if kind == "open":
                    await ws.send_json({"type": "loading"})
                    reply = await work(controller.prepare, data)
                elif kind == "start":
                    reply = await work(controller.start)
                elif kind == "presented":
                    await work(controller.acknowledge, data, received)
                    continue
                elif kind == "input":
                    reply = await work(controller.step, data, received)
                elif kind == "save":
                    reply = await work(controller.save)
                elif kind == "abort":
                    reply = await work(controller.abort, "user_aborted_or_focus_lost")
                else:
                    raise ValueError("Unknown control message")
                await ws.send_json(reply)
        except asyncio.TimeoutError:
            if not ws.closed:
                await ws.send_json({"type": "error", "message": "操作连接超时，本段已停止并排除训练。请重新连接。"})
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            if not ws.closed:
                await ws.send_json({"type": "error", "message": str(exc)[:600]})
        finally:
            try:
                await shutdown_controller(controller, executor)
            finally:
                holder["ws"] = None
                await ws.close()
        return ws

    async def static(request):
        name = request.match_info.get("name", "index.html")
        if name not in ("index.html", "app.js", "style.css"):
            raise web.HTTPNotFound()
        return web.FileResponse(Path(__file__).parent / "web_static" / name)

    async def shutdown(_app):
        if holder["ws"] is not None:
            await holder["ws"].close(code=1001, message=b"Server stopping")
    app.on_shutdown.append(shutdown)
    app.router.add_get("/", static)
    app.router.add_get("/static/{name}", static)
    app.router.add_post("/api/login", login)
    app.router.add_get("/api/catalog", catalog)
    app.router.add_get("/ws", websocket)
    return app


def serve(args):
    try:
        from aiohttp import web
    except ImportError as exc:
        raise RuntimeError("Install browser dependencies: pip install -e '.[web]'") from exc
    origins = args.origin or [f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"]
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.origin:
        raise ValueError("For a non-loopback bind, set --origin to the browser's exact HTTPS origin")
    for origin in origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("Origin must be an exact http(s)://host:port without a path")
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote browser origins require HTTPS; localhost works through an SSH tunnel")
    if not args.allow_software:
        from .cloud import graphics_check
        result = graphics_check()
        if not result["ok"]:
            raise RuntimeError("NVIDIA display is not ready. Use scripts/run_cloud.sh web, or --allow-software for diagnostics only.\n" + result["output"])
        os.environ["VALORANT_LAB_GPU_REQUIRED"] = "1"
    settings = Settings(output=Path(args.output), queue=Path(args.queue), state_dir=Path(args.state_dir),
                        max_frame_age_ms=args.max_frame_age_ms, control_timeout=args.control_timeout)
    token = token_file(args.token_file)
    print(f"访问口令文件：{Path(args.token_file).resolve()}（不要放进 URL 或 Git）", flush=True)
    print("浏览器示范使用逐帧确认的步进模式；实际游戏原速性能需单独 evaluate --realtime。", flush=True)
    app = create_app(settings, token=token, origins=origins)
    web.run_app(app, host=args.host, port=args.port, access_log=None, shutdown_timeout=90)
