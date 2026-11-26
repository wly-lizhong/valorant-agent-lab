"use strict";
const el = id => document.getElementById(id);
const canvas = el("game"), ctx = canvas.getContext("2d", {alpha: false});
let socket, tasks = [], state = "idle", timer, generation = 0;
let keys = new Set(), fire = false, reload = false, dx = 0, dy = 0;
const allowedKeys = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft", "ControlLeft", "Space"]);
function status(text) { el("status").textContent = text; }
function error(text) { el("error").textContent = text; el("error").hidden = !text; }
function send(message) {
  if (socket?.readyState !== WebSocket.OPEN) throw new Error("控制连接已断开，请重新加载场景。");
  socket.send(JSON.stringify(message));
}
function clearInput() { keys.clear(); fire = reload = false; dx = dy = 0; clearTimeout(timer); }
function release() { clearInput(); if (document.pointerLockElement === canvas) document.exitPointerLock(); }
function controls() {
  const busy = !["idle", "saved"].includes(state);
  for (const id of ["open", "task", "split", "seed", "mouse-scale", "refresh"]) el(id).disabled = busy;
  if (el("task").value.startsWith("case:")) { el("split").disabled = true; el("seed").disabled = true; }
  el("start").disabled = state !== "ready";
  el("stop").disabled = !busy || state === "loading";
}
function stop() {
  if (!["active", "starting", "ready", "review"].includes(state)) return;
  generation++; state = "stopping"; release(); el("save").hidden = true; controls();
  try { send({type:"abort"}); } catch (e) { fail(e); }
  status("已停止操作，本段不会进入训练。可重新加载场景再试。");
}
function fail(e) {
  generation++; state = "idle"; release(); el("save").hidden = true; controls();
  error(e.message || String(e)); status("本次已停止。重新加载场景即可再试。");
  socket?.close();
}
function describe() {
  const task = tasks.find(t => t.id === el("task").value);
  el("task-description").textContent = task?.description || "";
  const names = {pending:"等待示范", demonstrated:"已有示范，等待训练与复测", model_passed_once:"模型已通过一次，仍需回归验证"};
  el("case-status").textContent = task?.kind === "correction" ? `${names[task.status] || task.status} · 出现 ${task.occurrences} 次` : "";
  if (task?.kind === "correction") el("split").value = "train";
  controls();
}
async function catalog() {
  const response = await fetch("/api/catalog");
  if (!response.ok) throw new Error(response.status === 401 ? "登录已过期，请刷新页面重新连接。" : await response.text());
  const data = await response.json(), selected = el("task").value;
  tasks = data.tasks;
  el("task").replaceChildren();
  for (const [kind, title] of [["course", "基础课程"], ["correction", "系统生成的纠错任务"]]) {
    const group = document.createElement("optgroup"); group.label = title;
    for (const task of tasks.filter(t => t.kind === kind)) { const option = new Option(task.title, task.id); group.append(option); }
    if (group.children.length) el("task").append(group);
  }
  if (tasks.some(t => t.id === selected)) el("task").value = selected;
  el("test-mode").hidden = data.source === "human";
  el("login-panel").hidden = true; el("workspace").hidden = false;
  el("connection").textContent = data.busy && !socket ? "服务器有会话进行中" : "已登录";
  describe();
}
function connect() {
  if (socket?.readyState === WebSocket.OPEN) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`);
    socket = ws;
    ws.addEventListener("open", () => { el("connection").textContent = "控制已连接"; resolve(); });
    ws.addEventListener("error", () => reject(new Error("无法连接控制服务，可能有其他会话正在操作。")));
    ws.addEventListener("message", event => { if (socket === ws) receive(JSON.parse(event.data)).catch(fail); });
    ws.addEventListener("close", () => {
      if (socket !== ws) return;
      generation++; const interrupted = !["idle", "saved"].includes(state);
      state = "idle"; release(); el("save").hidden = true; controls(); el("connection").textContent = "控制已断开";
      if (interrupted) status("连接已中断，本段不会进入训练。请重新加载场景。");
    });
  });
}
async function receive(message) {
  if (message.type === "error") throw new Error(message.message);
  if (message.type === "loading") { status("服务器正在加载地图，请稍候……"); return; }
  if (message.type === "saved") {
    generation++; state = "saved"; release(); controls(); el("save").hidden = true;
    status(message.accepted ? `示范已提交：${message.episode}。等待下一次训练使用。` : "记录已保存供检查，未通过质量要求，不会进入训练。");
    await catalog(); return;
  }
  if (message.type === "stopped") { state = "idle"; controls(); return; }
  if (message.type !== "frame" || state === "stopping") return;
  const version = generation;
  const bytes = Uint8Array.from(atob(message.png), c => c.charCodeAt(0));
  const bitmap = await createImageBitmap(new Blob([bytes], {type:"image/png"}));
  if (version !== generation) { bitmap.close(); return; }
  canvas.width = message.width; canvas.height = message.height;
  ctx.drawImage(bitmap, 0, 0); bitmap.close(); el("placeholder").hidden = true;
  // Two animation frames bracket a paint opportunity. This is browser telemetry, not proof of physical display.
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  if (version !== generation) return;
  state = message.state; controls(); el("steps").textContent = message.step;
  el("age").textContent = message.frame_age_ms === undefined ? "—" : `${message.frame_age_ms.toFixed(0)} ms`;
  el("engine").textContent = message.engine_step_ms === undefined ? "—" : `${message.engine_step_ms.toFixed(0)} ms`;
  el("quality").textContent = message.quality_ok ? "画面与操作正常对应。完成后请确认提交。" : "本段出现明显延迟，已排除训练；完成后可保存供检查。";
  if (state === "ready") { status("场景已就绪。点击“开始操作并录制”，鼠标将锁定到游戏画面。"); return; }
  if (state === "review") {
    release(); el("save").hidden = false;
    el("save").textContent = message.success && message.quality_ok ? "确认提交成功示范" : "保存本次记录（不进入训练）";
    status(message.success ? "目标完成。确认这次操作是你希望智能体学习的示范后，再提交。" : "本次未完成目标。可以保存检查，或放弃后重新练习。");
    return;
  }
  if (state !== "active") return;
  if (document.pointerLockElement !== canvas || document.hidden || !document.hasFocus()) { stop(); return; }
  status("正在录制 · 完成任务后自动结束；Esc 退出会放弃本段。");
  // Drop motion collected while the previous image was still visible. Held keys remain held.
  dx = dy = 0;
  const displayed = performance.now();
  send({type:"presented", frame_id:message.frame_id, displayed_ms:displayed});
  clearTimeout(timer);
  timer = setTimeout(() => {
    if (version !== generation || state !== "active") return;
    if (document.pointerLockElement !== canvas || document.hidden || !document.hasFocus()) { stop(); return; }
    try { send({type:"input", frame_id:message.frame_id, sampled_ms:performance.now(), keys:[...keys],
      dx, dy, fire, reload, focused:true}); } catch (e) { fail(e); }
    dx = dy = 0;
  }, message.period_ms);
}
el("login-form").addEventListener("submit", async event => {
  event.preventDefault(); error("");
  try { const response = await fetch("/api/login", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({token:el("token").value})});
    if (!response.ok) throw new Error(await response.text()); el("token").value = ""; await catalog();
  } catch (e) { error(e.message); }
});
el("refresh").addEventListener("click", () => catalog().catch(e => error(e.message)));
el("task").addEventListener("change", describe);
el("open").addEventListener("click", async () => {
  error("");
  const seed = Number(el("seed").value), scale = Number(el("mouse-scale").value);
  if (!Number.isInteger(seed) || seed < 0 || seed >= 100000 || !Number.isFinite(scale) || scale < .001 || scale > 2) { error("请检查场景编号和鼠标灵敏度。"); return; }
  state = "loading"; generation++; controls(); el("save").hidden = true;
  try { await connect(); send({type:"open", task:el("task").value, seed, split:el("split").value, mouse_scale:scale}); }
  catch (e) { fail(e); }
});
el("start").addEventListener("click", async () => {
  try {
    if (!canvas.requestPointerLock) throw new Error("此浏览器不支持鼠标锁定，请使用桌面版 Chrome、Edge 或 Firefox。");
    state = "starting"; controls(); canvas.focus(); await canvas.requestPointerLock();
    // Older browsers return void; pointerlockchange will start recording in that case.
    if (document.pointerLockElement === canvas) begin();
  } catch (e) { state = "ready"; controls(); error(`无法锁定鼠标：${e.message}`); }
});
function begin() { if (state === "starting") { state = "active"; clearInput(); send({type:"start"}); } }
document.addEventListener("pointerlockchange", () => { if (document.pointerLockElement === canvas) begin(); else if (["active", "starting"].includes(state)) stop(); });
document.addEventListener("pointerlockerror", () => { if (state === "starting") { state = "ready"; controls(); error("鼠标锁定失败，请再次点击开始，或使用桌面浏览器。"); } });
el("stop").addEventListener("click", stop);
el("save").addEventListener("click", () => { try { el("save").hidden = true; send({type:"save"}); } catch (e) { fail(e); } });
document.addEventListener("keydown", event => {
  if (state !== "active" || document.pointerLockElement !== canvas) return;
  if (allowedKeys.has(event.code)) { keys.add(event.code); event.preventDefault(); }
  if (event.code === "KeyR") { if (!event.repeat) reload = true; event.preventDefault(); }
});
document.addEventListener("keyup", event => { keys.delete(event.code); if (event.code === "KeyR") reload = false; });
document.addEventListener("mousemove", event => { if (state === "active" && document.pointerLockElement === canvas) { dx += event.movementX; dy += event.movementY; } });
document.addEventListener("mousedown", event => { if (state === "active" && document.pointerLockElement === canvas && event.button === 0) fire = true; });
document.addEventListener("mouseup", event => { if (event.button === 0) fire = false; });
canvas.addEventListener("contextmenu", event => event.preventDefault());
window.addEventListener("blur", () => { if (["active", "starting"].includes(state)) stop(); });
document.addEventListener("visibilitychange", () => { if (document.hidden && ["active", "starting"].includes(state)) stop(); });
window.addEventListener("pagehide", () => { socket?.close(); });
