"""Fixed-course launcher for the standalone realtime trial desktop."""
import asyncio
import json
import os
from pathlib import Path

COURSES = {
    'lotus_easy': dict(title='01 · 莲华古城：入门清点', mode='siteclear', tier=0,
        hint='沿青色引导线向 A 点推进，观察转角，停下后瞄准敌人开枪。入门难度每轮最多 2 名守点敌人。'),
    'lotus_normal': dict(title='02 · 莲华古城：进阶清点', mode='siteclear', tier=2,
        hint='同一张真实地图，提高守点难度；每轮最多 3 名敌人。逐个检查转角，避免边移动边连续开枪。'),
    'stop_shoot': dict(title='03 · 训练场：急停点射', mode='counterstrafe', tier=0,
        hint='按 A / D 左右移动，看到靶子后停下并点射。这一关是专用训练场，不是莲华古城地图。'),
}

class GameRunner:
    def __init__(self, root=None, runtime=None):
        self.root = Path(root or os.environ.get('VALORANT_LAB_ROOT')
                         or Path(__file__).resolve().parents[2]).resolve()
        self.runtime = Path(runtime or self.root / 'runs/desktop-trial')
        self.process = None

    async def stop(self):
        p, self.process = self.process, None
        if p is not None and p.returncode is None:
            p.terminate()
            try:
                await asyncio.wait_for(p.wait(), 5)
            except asyncio.TimeoutError:
                p.kill()
                await p.wait()

    async def start(self, course):
        if course not in COURSES:
            raise ValueError('请选择列表中的关卡')
        await self.stop()
        cfg = json.loads((self.root / 'local.json').read_text())
        item = COURSES[course]
        args = [cfg['godot'], '--path', cfg['project'], '--rendering-method',
                'gl_compatibility', '--audio-driver', 'Dummy', '--resolution',
                '1280x720', '--windowed', '--print-fps', '--', '--map', 'lotus',
                '--mode', item['mode'], '--tier', str(item['tier'])]
        log = self.runtime / 'selected-game.log'
        self.runtime.mkdir(parents=True, exist_ok=True)
        with log.open('wb') as output:
            self.process = await asyncio.create_subprocess_exec(*args, stdout=output,
                stderr=asyncio.subprocess.STDOUT, cwd=self.root)
        try:
            for _ in range(240):
                if self.process.returncode is not None:
                    raise RuntimeError('游戏启动失败')
                if 'Project FPS:' in log.read_text(errors='replace'):
                    return
                await asyncio.sleep(.25)
            raise RuntimeError('场景加载超时')
        except BaseException:
            await self.stop()
            raise
