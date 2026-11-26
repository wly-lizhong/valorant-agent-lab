// Run with: node --test tests/web_guidance.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.join(__dirname, '..', 'src', 'valorant_lab', 'web_static');

function harness() {
  const ids = [...fs.readFileSync(path.join(root, 'index.html'), 'utf8').matchAll(/id="([^"]+)"/g)].map(m => m[1]);
  const context = new Proxy({}, {get: (o, k) => o[k] || (() => {})});
  const nodes = Object.fromEntries(ids.map(id => [id, {hidden:false, value:'', width:260, height:200,
    textContent:'', addEventListener(){}, getContext(){return context;}}]));
  nodes['guide-mode'].value = 'practice';
  const sandbox = {document:{getElementById:id=>nodes[id], addEventListener(){}},
    window:{addEventListener(){}}, clearTimeout, setTimeout, console};
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(root, 'app.js'), 'utf8'), sandbox);
  return {nodes, run: code => vm.runInContext(code, sandbox)};
}

test('practice shows progress, direction and endpoint, missing telemetry clears stale map', () => {
  const {nodes, run} = harness();
  run(`drawGuide({route:[[0,0,0],[0,0,-4]],position:[0,0,0],yaw:0,
    completed:0,total:1,distance:4,radius:1.1,angle:0,instruction:'按 W 前进',success:false})`);
  assert.equal(nodes['route-guide'].hidden, false);
  assert.equal(nodes['guide-hud'].hidden, false);
  assert.match(nodes['guide-hud'].textContent, /↑.*4.0 米/);
  assert.match(nodes['route-distance'].textContent, /1 号（终点）/);
  run('drawGuide(null)');
  assert.equal(nodes['guide-hud'].hidden, true);
  assert.match(nodes['route-instruction'].textContent, /没有定位/);
  assert.equal(nodes['route-distance'].textContent, '');
});

test('formal recording hides guide and modes cannot change during an episode', () => {
  const {nodes, run} = harness();
  run('guided=false; drawGuide(null); state="active"; controls()');
  assert.equal(nodes['route-guide'].hidden, true);
  assert.equal(nodes['guide-hud'].hidden, true);
  assert.equal(nodes['guide-mode'].disabled, true);
  run('state="saved"; controls()');
  assert.equal(nodes['guide-mode'].disabled, false);
});

test('completion and missing waypoint have distinct prominent messages', () => {
  const {nodes, run} = harness();
  run('guided=false; showOutcome({quality_ok:true,outcome:{success:true,completed:5,total:5}})');
  assert.match(nodes['result-title'].textContent, /任务完成/);
  assert.match(nodes['result-detail'].textContent, /确认提交/);
  run('showOutcome({outcome:{success:false,completed:2,total:5}}, true)');
  assert.match(nodes['result-title'].textContent, /已停止/);
  assert.match(nodes['result-detail'].textContent, /下一个是 3 号点/);
  run('showOutcome({outcome:{success:false,completed:2,total:5,reason:"timeout"}})');
  assert.match(nodes['result-detail'].textContent, /时间已用完/);
});
