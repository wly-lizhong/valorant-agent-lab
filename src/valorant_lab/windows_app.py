"""Chinese desktop launcher. Tk owns the UI thread; capture/upload run on a worker."""
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from .env import ROOT
from .native_capture import record_native
from .corrections import load_case
from .exchange import pull, push, DEFAULT_REPO

COURSES = [
    ("路线行走", "lotus_route", "只走路：从 A Root 沿通道走向 A 入口，依次通过路线点。"),
    ("停稳射击", "lotus_shooting", "沿路线前进，停稳后处理可见敌人，再继续到入口。"),
    ("换弹与前进", "lotus_reload", "先换弹，再处理敌人并完成路线。"),
    ("调整视角", "lotus_recovery", "将视角恢复到正常高度，再完成路线。"),
    ("组合交战", "lotus_movement_combat", "移动、停稳射击、换弹并完成路线。"),
]
ISSUES = {"window_lost_focus": "切换了窗口", "render_or_disk_too_slow": "本机出现明显卡顿",
          "writer_queue_full": "硬盘保存跟不上", "capture_failed": "获取画面失败",
          "user_aborted": "主动放弃", "window_closed": "关闭了游戏窗口"}


def main():
    root = tk.Tk(); root.title("Agent Lab · Windows 本机示范台")
    root.geometry("840x600"); root.minsize(800, 580)
    panel = ttk.Frame(root, padding=24); panel.pack(fill="both", expand=True)
    ttk.Label(panel, text="本机玩游戏，云端训练", font=("Microsoft YaHei", 20, "bold")).pack(anchor="w")
    ttk.Label(panel, text="选择课程或云端发来的纠错题。游戏、画面和按键全部在这台电脑运行。").pack(anchor="w", pady=10)
    selected = tk.StringVar(); picker = ttk.Combobox(panel, textvariable=selected, state="readonly", width=65)
    picker.pack(fill="x", pady=8)
    hint = tk.StringVar(); ttk.Label(panel, textvariable=hint, wraplength=710).pack(anchor="w", pady=8)
    row = ttk.Frame(panel); row.pack(fill="x", pady=8)
    ttk.Label(row, text="场景编号").pack(side="left")
    seed = tk.StringVar(value="42"); seed_input = ttk.Entry(row, textvariable=seed, width=9); seed_input.pack(side="left", padx=8)
    split = tk.StringVar(value="训练示范")
    split_box = ttk.Combobox(row, textvariable=split, values=["训练示范", "独立验证"], state="readonly", width=12)
    split_box.pack(side="left"); ttk.Label(row, text="验证集不用于学习；纠错题固定用于训练。").pack(side="left", padx=8)
    controls = ttk.Frame(panel); controls.pack(fill="x", pady=12)
    status = tk.StringVar(value="准备就绪。先选关，再点击开始。")
    ttk.Label(panel, textvariable=status, wraplength=710, foreground="#175b55").pack(anchor="w", pady=8)
    ttk.Separator(panel).pack(fill="x", pady=12)
    ttk.Label(panel, text="GitHub 数据交换", font=("Microsoft YaHei", 12, "bold")).pack(anchor="w")
    repo = tk.StringVar(value=DEFAULT_REPO)
    ttk.Entry(panel, textvariable=repo).pack(fill="x", pady=6)
    ttk.Label(panel, text="当前仓库公开：上传的游戏画面、操作和题目附件也会公开。不会上传密码、桌面或配置文件。",
              wraplength=710).pack(anchor="w")
    transfers = ttk.Frame(panel); transfers.pack(fill="x", pady=12)
    ttk.Label(panel, text="游戏中 Enter 开始；成功后 Enter 确认保存；Esc 放弃返回。\n上传需要 GitHub CLI（gh）已登录。首次请运行 gh auth login。",
              wraplength=710).pack(anchor="w", pady=8)
    results = queue.Queue(); busy = False; items = []; buttons = []

    def refresh():
        items.clear()
        for title, key, description in COURSES:
            items.append((title, ROOT / "configs" / (key + ".json"), None, description))
        for path in sorted((ROOT / "runs/corrections-v215").glob("*/case.json")):
            try:
                c = load_case(path)
            except ValueError:
                continue  # Older-engine tasks remain on disk; do not relabel them as current tasks.
            if c.get("status") != "model_passed_once":
                items.append(("纠错 · " + c["title"] + " · " + c["id"][:6], None, path,
                              c["note"] + " 需要完整重做，不是直接从卡住的一帧开始。"))
        picker["values"] = [x[0] for x in items]; picker.current(0); describe()

    def describe(*_):
        if picker.current() >= 0:
            _, _, case, description = items[picker.current()]
            hint.set(description)
            seed_input.configure(state="disabled" if case else "normal")
            split_box.configure(state="disabled" if case else "readonly")

    def work(function, message):
        nonlocal busy
        if busy: return
        busy = True; status.set(message)
        for b in buttons: b.configure(state="disabled")
        picker.configure(state="disabled")
        def target():
            try: results.put((True, function()))
            except Exception as e: results.put((False, str(e)))
        threading.Thread(target=target, daemon=True).start()

    def record():
        try:
            number = int(seed.get())
            if not 0 <= number < 100000: raise ValueError()
        except ValueError:
            messagebox.showerror("场景编号", "请输入 0 到 99999 之间的整数。"); return
        _, scenario, case, _ = items[picker.current()]
        chosen_split = "train" if case or split.get() == "训练示范" else "validation"
        def capture():
            path = record_native(scenario=scenario, case=case, seed=number, split=chosen_split)
            m = json.loads((path / "episode.json").read_text(encoding="utf-8"))
            reasons = "、".join(ISSUES.get(x, "数据校验未通过") for x in m["quality_issues"])
            return "示范已保存，可上传供云端训练。" if m["accepted"] else "本次记录已保存供检查，未进入训练。" + reasons
        work(capture, "正在打开本机游戏。请在游戏窗口按 Enter 开始；结束后会自动回到这里。")

    def download():
        chosen_repo = repo.get().strip()
        def receive():
            result = pull(ROOT / "runs/corrections-v215", "tasks", chosen_repo)
            return f"题目同步完成：新增 {result['imported']} 题，已有 {result['skipped']} 题。"
        work(receive, "正在从 GitHub 下载尚未接收的纠错题包…")

    def upload():
        chosen_repo = repo.get().strip()
        work(lambda: "已上传示范批次：" + push(ROOT / "runs/demonstrations-v215", "demos", chosen_repo), "正在校验并上传成功示范附件…")

    for parent, text, action in [(controls, "开始本机录制", record), (controls, "刷新题目", refresh),
                                  (transfers, "下载云端纠错题", download), (transfers, "上传成功示范", upload)]:
        b = ttk.Button(parent, text=text, command=action); b.pack(side="left", padx=(0, 12)); buttons.append(b)
    picker.bind("<<ComboboxSelected>>", describe)

    def poll():
        nonlocal busy
        try:
            ok, value = results.get_nowait(); busy = False
            for b in buttons: b.configure(state="normal")
            picker.configure(state="readonly")
            if ok: refresh(); status.set(str(value))
            else: status.set("操作失败：" + value); messagebox.showerror("操作失败", value)
        except queue.Empty: pass
        root.after(150, poll)

    def close():
        if busy: messagebox.showinfo("任务进行中", "请先在游戏中按 Esc 返回，或等待当前传输结束。")
        else: root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    refresh(); poll(); root.mainloop()


if __name__ == "__main__": main()
