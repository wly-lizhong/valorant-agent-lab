"use strict";
const el = id => document.getElementById(id);
const canvas = el("game"), ctx = canvas.getContext("2d", {alpha: false});
let socket, tasks = [], state = "idle", timer, generation = 0;
let keys = new Set(), fire = false, reload = false, dx = 0, dy = 0;
let guided = true;
function drawGuide(guide) {
  el("route-guide").hidden = !guided;
  el("guide-hud").hidden = !guided || !guide;
  if (!guided) return;
  const map = el("route-map"), c = map.getContext("2d");
  c.clearRect(0, 0, map.width, map.height);
  if (!guide) {
    el("route-instruction").textContent = "暂时没有定位信息，请重新加载场景";
    el("route-progress").textContent = el("route-distance").textContent = "";
    el("route-progress-bar").value = 0;
    return;
  }
  el("route-instruction").textContent = guide.instruction;
  el("route-progress").textContent = `路线点：已完成 ${guide.completed} / ${guide.total}`;
  el("route-progress-bar").max = guide.total;
  el("route-progress-bar").value = guide.completed;
  el("route-distance").textContent = guide.success ? "已到终点，可以结束练习或重新加载后正式录制。" : `下一点：${Math.min(guide.completed + 1, guide.total)} 号${guide.completed >= guide.total - 1 ? "（终点）" : ""} · 距离 ${guide.distance.toFixed(1)} 米 · 判定半径 ${guide.radius.toFixed(1)} 米`;
  const arrow = Math.abs(guide.angle) > 150 ? "↶" : guide.angle > 15 ? "→" : guide.angle < -15 ? "←" : "↑";
  el("guide-hud").textContent = guide.success ? "✓ 已到终点" : `${arrow} ${guide.instruction} · ${guide.distance.toFixed(1)} 米`;
  const points = [...guide.route, guide.position];
  const xs = points.map(p => p[0]), zs = points.map(p => p[2]);
  const minX = Math.min(...xs) - 2, maxX = Math.max(...xs) + 2;
  const minZ = Math.min(...zs) - 2, maxZ = Math.max(...zs) + 2;
  const scale = Math.min(220 / (maxX - minX), 160 / (maxZ - minZ));
  const project = p => [130 + (p[0] - (minX + maxX) / 2) * scale, 100 + (p[2] - (minZ + maxZ) / 2) * scale];
  const route = guide.route.map(project);
  c.strokeStyle = "#6d879e"; c.lineWidth = 3; c.beginPath();
  route.forEach(([x,y], i) => i ? c.lineTo(x,y) : c.moveTo(x,y)); c.stroke();
  c.font = "bold 13px sans-serif";
  route.forEach(([x,y], i) => {
    c.fillStyle = i === guide.completed + 1 ? "#ffcf60" : i <= guide.completed ? "#6be3b8" : "#8da7bd";
    c.beginPath(); c.arc(x,y,5,0,Math.PI*2); c.fill();
    c.fillText(i === 0 ? "起点" : `${i}${i === guide.total ? " 终点" : ""}`, x+9,y+4);
  });
  const [gx,gy] = route[route.length-1]; c.strokeStyle="#6be3b8"; c.lineWidth=2;
  c.beginPath(); c.arc(gx,gy,Math.max(8,guide.radius*scale),0,Math.PI*2); c.stroke();
  const [px,py] = project(guide.position); c.save(); c.translate(px,py); c.rotate(guide.yaw*Math.PI/180);
  c.fillStyle="#fff"; c.strokeStyle="#101820"; c.lineWidth=2;
  c.beginPath(); c.moveTo(0,-10); c.lineTo(7,8); c.lineTo(0,4); c.lineTo(-7,8); c.closePath(); c.fill(); c.stroke(); c.restore();
}
const allowedKeys = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft", "ControlLeft", "Space"]);
function status(text) { el("status").textContent = text; }
function showOutcome(message, stopped = false) {
  const outcome = message.outcome || {}, success = outcome.success === true;
  el("result-banner").hidden = false;
  el("result-title").textContent = stopped ? "本次已停止 · 未提交示范" : success ? "✓ 任务完成" : "本次结束 · 任务未完成";
  const reason = {timeout:"本次练习时间已用完。", death:"角色已死亡。", fall:"角色跌落，任务结束。"}[outcome.reason] || "";
  const progress = outcome.total ? `路线点已通过 ${outcome.completed} / ${outcome.total}。` : "";
  const missing = outcome.total && outcome.completed < outcome.total ? `需要依次经过全部路线点；下一个是 ${outcome.completed + 1} 号点，走到后面的区域不会自动补过漏点。` : "";
  el("result-detail").textContent = reason + progress + (success ? stopped ? "本次未提交，可重新加载再试。" : guided ? "练习完成，可结束练习后切换正式示范。" : message.quality_ok ? "请点击下方按钮确认提交示范。" : "任务成功，但记录未通过质量检查，可保存供检查。" : missing || "请根据本课要求完成路线与交战，再重试。 ");
}
function error(text) { el("error").textContent = text; el("error").hidden = !text; }
function send(message) {
  if (socket?.readyState !== WebSocket.OPEN) throw new Error("控制连接已断开，请重新加载场景。");
  socket.send(JSON.stringify(message));
}
function clearInput() { keys.clear(); fire = reload = false; dx = dy = 0; clearTimeout(timer); }
function release() { clearInput(); if (document.pointerLockElement === canvas) document.exitPointerLock(); }
function controls() {
  const busy = !["idle", "saved"].includes(state);
  for (const id of ["open", "task", "split", "seed", "mouse-scale", "refresh", "guide-mode"]) el(id).disabled = busy;
  el("start").textContent = el("guide-mode").value === "practice" ? "开始带引导练习" : "开始操作并录制";
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
    status(message.guided ? "练习已结束，没有写入训练数据。可选择正式示范并重新加载场景。" : message.accepted ? `示范已提交：${message.episode}。等待下一次训练使用。` : "记录已保存供检查，未通过质量要求，不会进入训练。");
    await catalog(); return;
  }
  if (message.type === "stopped") { state = "idle"; controls(); showOutcome(message, true); return; }
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
  guided = message.guided === true; drawGuide(message.guidance);
  el("age").textContent = message.frame_age_ms === undefined ? "—" : `${message.frame_age_ms.toFixed(0)} ms`;
  el("engine").textContent = message.engine_step_ms === undefined ? "—" : `${message.engine_step_ms.toFixed(0)} ms`;
  el("quality").textContent = guided ? "当前为带引导练习，不录制训练数据。" : !message.quality_ok ? "操作采样出现异常停顿，已排除训练；完成后可保存供检查。" : message.latency_warning ? "画面往返较慢，游戏会减速；当前画面与动作仍对应，未因此排除训练。" : "画面与操作正常对应。完成后请确认提交。";
  if (state === "ready") { status(`场景已就绪。点击“${el("start").textContent}”，鼠标将锁定到游戏画面。`); return; }
  if (state === "review") {
    release(); el("save").hidden = false;
    showOutcome(message);
    el("result-banner").scrollIntoView({block:"nearest"});
    if (guided) {
      el("save").textContent = "结束练习";
      status(message.success ? "练习完成！结束练习后，可切换正式示范并重新加载。" : "练习结束，尚未完成目标。结束后重新加载再试。");
      return;
    }
    el("save").textContent = message.success && message.quality_ok ? "确认提交成功示范" : "保存本次记录（不进入训练）";
    status(message.success ? "目标完成。确认这次操作是你希望智能体学习的示范后，再提交。" : "本次未完成目标。可以保存检查，或放弃后重新练习。");
    return;
  }
  if (state !== "active") return;
  if (document.pointerLockElement !== canvas || document.hidden || !document.hasFocus()) { stop(); return; }
  status(guided ? "正在练习 · 跟随上方黄色路线点；Esc 退出。本次不会录入训练。" : "正在录制 · 完成任务后自动结束；Esc 退出会放弃本段。");
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
el("guide-mode").addEventListener("change", () => { el("route-guide").hidden = el("guide-hud").hidden = true; controls(); });
el("open").addEventListener("click", async () => {
  error("");
  const seed = Number(el("seed").value), scale = Number(el("mouse-scale").value);
  if (!Number.isInteger(seed) || seed < 0 || seed >= 100000 || !Number.isFinite(scale) || scale < .001 || scale > 2) { error("请检查场景编号和鼠标灵敏度。"); return; }
  state = "loading"; generation++; controls(); el("save").hidden = true;
  el("result-banner").hidden = true;
  el("route-guide").hidden = el("guide-hud").hidden = true;
  try { await connect(); send({type:"open", task:el("task").value, seed, split:el("split").value, mouse_scale:scale, guided:el("guide-mode").value === "practice"}); }
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
