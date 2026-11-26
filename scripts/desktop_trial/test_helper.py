import asyncio
from types import SimpleNamespace
import pytest
from aiohttp.test_utils import TestClient,TestServer
from server import make_app,Controls
from game_runner import COURSES,GameRunner


class Fake:
    instances=[]
    def __init__(self): self.calls=[];self.closed=False;self.instances.append(self)
    def apply(self,data): self.calls.append(data)
    def release(self): self.calls.append({'type':'release'})
    def close(self): self.closed=True


def test_origin_exclusivity_disconnect_cleanup(tmp_path):
    async def run():
        headers={'Host':'127.0.0.1:16080','Origin':'http://127.0.0.1:16080'}
        async with TestClient(TestServer(make_app(Fake,tmp_path))) as client:
            assert (await client.get('/input',headers={**headers,'Origin':'http://evil.test'})).status==403
            ws=await client.ws_connect('/input',headers=headers)
            assert (await ws.receive_json())['ready']
            assert (await client.get('/input',headers=headers)).status==409
            # A background-tab timer delay must release held inputs, not tear
            # down the connection and require a reload after just two seconds.
            await asyncio.sleep(2.15)
            assert {'type':'release'} in Fake.instances[-1].calls
            assert not ws.closed
            await ws.send_json({'type':'ping'})
            await ws.close()
            for _ in range(30):
                if Fake.instances[-1].closed: break
                await asyncio.sleep(.01)
            assert Fake.instances[-1].closed
    asyncio.run(run())


def test_control_validation_and_release():
    c=Controls.__new__(Controls)
    calls=[]
    c.x=SimpleNamespace(XFlush=lambda *a:None)
    c.t=SimpleNamespace(XTestFakeRelativeMotionEvent=lambda *a:calls.append(a),
                        XTestFakeKeyEvent=lambda *a:calls.append(a),
                        XTestFakeButtonEvent=lambda *a:calls.append(a))
    c.display=1;c.codes={'KeyW':25};c.keys=set();c.buttons=set();c.motion_log_count=0
    for data in [{'type':'move','dx':float('nan'),'dy':1}, {'type':'move','dx':1001,'dy':1},
                 {'type':'key','key':'exec','down':True}, {'type':'button','button':1,'down':1}]:
        with pytest.raises(ValueError):c.apply(data)
    assert not calls
    c.apply({'type':'move','dx':12,'dy':-5})
    assert calls[-1]==(1,12,-5,0)
    c.apply({'type':'key','key':'KeyW','down':True})
    c.apply({'type':'button','button':1,'down':True})
    c.release()
    assert not c.keys and not c.buttons
    assert (1,25,0,0) in calls and (1,1,0,0) in calls


def test_course_start_cancel_and_disconnect(tmp_path):
    class Runner:
        def __init__(self): self.started=[];self.stops=0;self.gate=asyncio.Event()
        async def start(self, course): self.started.append(course);await self.gate.wait()
        async def stop(self): self.stops+=1
    async def run():
        runner=Runner()
        headers={'Host':'127.0.0.1:16080','Origin':'http://127.0.0.1:16080'}
        async with TestClient(TestServer(make_app(Fake,tmp_path,runner))) as client:
            catalog=await (await client.get('/courses',headers=headers)).json()
            assert {c['id'] for c in catalog}==set(COURSES)
            ws=await client.ws_connect('/input',headers=headers)
            await ws.receive_json()
            await ws.send_json({'type':'start','course':'../../shell'})
            assert 'error' in await ws.receive_json()
            assert not runner.started
            await ws.send_json({'type':'start','course':'lotus_easy'})
            await ws.send_json({'type':'key','key':'KeyW','down':True})
            await ws.send_json({'type':'stop'})
            assert (await ws.receive_json())['stopped']
            assert not any(c.get('key')=='KeyW' for c in Fake.instances[-1].calls)
            runner.gate.set()
            await ws.send_json({'type':'start','course':'lotus_normal'})
            assert (await ws.receive_json())['launched']
            await ws.send_json({'type':'move','dx':12,'dy':-5})
            await ws.close()
            await asyncio.sleep(.05)
            assert {'type':'move','dx':12,'dy':-5} in Fake.instances[-1].calls
            assert runner.stops>=3
    asyncio.run(run())


def test_launcher_rejects_unknown_course_before_touching_runtime(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(GameRunner(tmp_path).start('not-allowed'))
