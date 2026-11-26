"""Loopback-only desktop helper: noVNC video and relative X11 game controls.

Operational utility, not a demonstration recorder. Access only through SSH.
"""
import asyncio
import ctypes as C
import json
import time
from pathlib import Path
from aiohttp import web, WSMsgType
from game_runner import GameRunner, COURSES

KEYS = {**{f"Key{k}": k.lower() for k in "WASDRQEF"}, "Space": "space",
        "ShiftLeft": "Shift_L", "ControlLeft": "Control_L", "Escape": "Escape",
        "Digit1": "1", "Digit2": "2", "Digit3": "3", "Tab": "Tab"}


class Controls:
    def __init__(self):
        self.x = C.CDLL("libX11.so.6")
        self.t = C.CDLL("libXtst.so.6")
        self.x.XOpenDisplay.argtypes = [C.c_char_p]
        self.x.XOpenDisplay.restype = C.c_void_p
        self.x.XStringToKeysym.argtypes = [C.c_char_p]
        self.x.XStringToKeysym.restype = C.c_ulong
        self.x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        self.x.XKeysymToKeycode.restype = C.c_uint
        self.x.XFlush.argtypes = [C.c_void_p]
        self.x.XCloseDisplay.argtypes = [C.c_void_p]
        self.t.XTestFakeRelativeMotionEvent.argtypes = [C.c_void_p,C.c_int,C.c_int,C.c_ulong]
        self.t.XTestFakeKeyEvent.argtypes = [C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
        self.t.XTestFakeButtonEvent.argtypes = [C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("Cannot open configured X display")
        self.keys, self.buttons = set(), set()
        self.codes = {k:self.x.XKeysymToKeycode(self.display,self.x.XStringToKeysym(v.encode())) for k,v in KEYS.items()}

    def apply(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected control object")
        kind = data.get("type")
        if kind == "move":
            dx, dy = data.get("dx"), data.get("dy")
            if any(type(v) is not int or abs(v) > 1000 for v in (dx,dy)):
                raise ValueError("Invalid mouse motion")
            self.t.XTestFakeRelativeMotionEvent(self.display, dx, dy, 0)
        elif kind == "key":
            key, down = data.get("key"), data.get("down")
            if key not in KEYS or type(down) is not bool:
                raise ValueError("Invalid key")
            if down != (key in self.keys):
                self.t.XTestFakeKeyEvent(self.display,self.codes[key],int(down),0)
                (self.keys.add if down else self.keys.discard)(key)
        elif kind == "button":
            button, down = data.get("button"), data.get("down")
            if type(button) is not int or button not in (1,3) or type(down) is not bool:
                raise ValueError("Invalid mouse button")
            self.t.XTestFakeButtonEvent(self.display,button,int(down),0)
            (self.buttons.add if down else self.buttons.discard)(button)
        elif kind == "release":
            self.release()
        elif kind != "ping":
            raise ValueError("Unknown control")
        self.x.XFlush(self.display)

    def release(self):
        for key in self.keys:
            self.t.XTestFakeKeyEvent(self.display,self.codes[key],0,0)
        for button in self.buttons:
            self.t.XTestFakeButtonEvent(self.display,button,0,0)
        self.keys.clear(); self.buttons.clear()
        self.x.XFlush(self.display)

    def close(self):
        self.release()
        self.x.XCloseDisplay(self.display)


def make_app(factory=Controls, novnc_dir="/usr/share/novnc/", runner=None):
    owner = None
    game = runner or GameRunner()
    @web.middleware
    async def guard(request, handler):
        if request.headers.get("Host") not in ("127.0.0.1:16080", "localhost:16080", "127.0.0.1:6080"):
            raise web.HTTPForbidden()
        if request.path in ("/input", "/rfb") and request.headers.get("Origin") not in (
            "http://127.0.0.1:16080", "http://localhost:16080"):
            raise web.HTTPForbidden()
        return await handler(request)
    app = web.Application(middlewares=[guard])
    async def controls(request):
        nonlocal owner
        if owner is not None:
            raise web.HTTPConflict(text="Another game control session is connected")
        ws = web.WebSocketResponse(max_msg_size=4096, heartbeat=10)
        owner = ws
        control = None
        launch = None
        playing = False
        try:
            control = factory()
            await ws.prepare(request)
            await ws.send_json({"ready":True})
            async def start_game(course):
                nonlocal playing
                try:
                    await game.start(course)
                    playing = True
                    await ws.send_json({"launched":True,"course":course})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f'launch_failed: {type(exc).__name__}', flush=True)
                    await ws.send_json({"error":"场景启动失败，请重试；持续失败请告诉我。"})
            async def stop_game():
                nonlocal launch, playing
                playing = False
                control.release()
                if launch is not None:
                    launch.cancel()
                    await asyncio.gather(launch, return_exceptions=True)
                    launch = None
                await game.stop()
            last_message = time.monotonic()
            while not ws.closed:
                try:
                    message = await ws.receive(timeout=2)
                except asyncio.TimeoutError:
                    control.release()
                    if time.monotonic() - last_message > 15:
                        break
                    continue
                last_message = time.monotonic()
                if message.type in (WSMsgType.CLOSE,WSMsgType.CLOSED,WSMsgType.ERROR): break
                if message.type != WSMsgType.TEXT: raise ValueError("Text controls required")
                data = json.loads(message.data)
                if not isinstance(data, dict): raise ValueError('Expected object')
                kind = data.get('type')
                if kind == 'start':
                    if data.get('course') not in COURSES:
                        await ws.send_json({'error':'请选择列表中的关卡'})
                        continue
                    await stop_game()
                    launch = asyncio.create_task(start_game(data['course']))
                elif kind == 'stop':
                    await stop_game()
                    await ws.send_json({'stopped':True})
                elif kind in ('release','ping') or playing:
                    control.apply(data)
        except (asyncio.TimeoutError, ValueError, TypeError):
            pass
        finally:
            if launch is not None:
                launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)
            await game.stop()
            if control is not None: control.close()
            owner = None
            await ws.close()
        return ws
    async def video(request):
        ws = web.WebSocketResponse(max_msg_size=2**20, protocols=("binary",))
        reader, writer = await asyncio.open_connection("127.0.0.1",5901)
        await ws.prepare(request)
        async def outbound():
            while data := await reader.read(262144):
                await ws.send_bytes(data)
            await ws.close()
        task = asyncio.create_task(outbound())
        try:
            async for message in ws:
                if message.type != WSMsgType.BINARY: break
                writer.write(message.data)
                await writer.drain()
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            writer.close()
            await writer.wait_closed()
        return ws
    async def index(_):
        return web.FileResponse(Path(__file__).with_name("index.html"))
    app.router.add_get("/",index)
    app.router.add_get("/input",controls)
    app.router.add_get("/rfb",video)
    async def courses(_):
        return web.json_response([{'id':key,'title':value['title'],'hint':value['hint']}
                                  for key,value in COURSES.items()])
    app.router.add_get('/courses',courses)
    async def shutdown(_):
        if owner is not None: await owner.close()
        await game.stop()
    app.on_shutdown.append(shutdown)
    app.router.add_static("/novnc/",novnc_dir,show_index=False,follow_symlinks=True)
    return app


if __name__ == "__main__":
    web.run_app(make_app(),host="127.0.0.1",port=6080,access_log=None)
