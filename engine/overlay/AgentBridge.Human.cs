using Godot;
using CaValAim.Core;
using System.Text.Json;
using System.IO.Compression;
using System.Threading.Channels;

namespace CaValAim.AgentLab;

// Local input and rendered pre-action frames share the training action contract.
// No network, global keyboard hooks, or Python frame round trips are involved.
public partial class AgentBridge
{
    readonly string? humanDirectory;
    string humanState = "ready";
    Label? humanMenu;
    NativeWriter? humanWriter;
    bool humanCapturing, humanReload, humanJumpReady;
    byte[]? humanPixels;
    object? humanBefore;
    Vector2 humanMouse;
    int[] humanAction = Array.Empty<int>();
    double humanSampleWall;
    double humanCaptureMs, humanLastFrameMs, humanRenderWaitMs, humanReadbackMs;
    double humanPausedSeconds, humanLastTitleWall;
    object? humanFinal;
    readonly Queue<double> humanRecoveries = new();
    int humanRecoveryCount, humanTestStallStep = -1;
    bool HumanTest => CmdLine.Has("--agent-human-test");
    int HumanTestSteps => int.TryParse(CmdLine.After("--agent-human-test-steps"), out int value) ? Math.Clamp(value, 1, config.MaxSteps) : 20;

    void HumanSetup()
    {
        if (Directory.Exists(humanDirectory)) throw new ArgumentException("Capture directory already exists");
        ResetEpisode(int.Parse(CmdLine.After("--agent-seed") ?? "42"));
        humanWriter = new NativeWriter(humanDirectory!);
        humanMenu = new Label {
            Text = "本机示范录制\n\n按 Enter 开始\nWASD 移动，鼠标瞄准，左键开枪\nR 换弹，Shift 静步，Ctrl 蹲，Space 跳跃\nEsc 放弃本次\n\n成功后按 Enter 确认保存；失败或放弃不会进入训练。",
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            MouseFilter = Control.MouseFilterEnum.Ignore
        };
        var humanFont = new SystemFont { FontNames = new[] { "Microsoft YaHei", "Noto Sans CJK SC", "sans-serif" } };
        humanMenu.AddThemeFontOverride("font", humanFont);
        humanMenu.AddThemeFontSizeOverride("font_size", 22);
        humanMenu.AddThemeColorOverride("font_shadow_color", Colors.Black);
        humanMenu.AddThemeConstantOverride("shadow_offset_x", 2);
        humanMenu.AddThemeConstantOverride("shadow_offset_y", 2);
        var layer = new CanvasLayer(); AddChild(layer); layer.AddChild(humanMenu);
        humanMenu.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        Input.MouseMode = Input.MouseModeEnum.Visible;
        GetWindow().Title = "本机录制 · 按 Enter 开始";
        try { HumanGuideSetup(humanFont); }
        catch (Exception e)
        {
            GD.PrintErr("[native-guide] startup failed: " + e.Message);
            humanWriter.Issues.Add("guide_failed"); HumanFinish(false); return;
        }
        GetWindow().FocusExited += () => { if (humanState == "active" && !HumanTest) HumanStop("window_lost_focus"); };
        if (HumanTest) CallDeferred(nameof(HumanBegin));
    }

    public override void _Input(InputEvent e)
    {
        if (humanDirectory == null) return;
        if (e is InputEventKey { Pressed: true, Echo: false } k)
        {
            if (k.Keycode == Key.Enter && humanState == "ready") HumanBegin();
            else if (k.Keycode == Key.Enter && humanState == "review" && success) HumanFinish(true);
            else if (k.Keycode == Key.Escape) { HumanStop("user_aborted"); HumanFinish(false); }
            else if (k.PhysicalKeycode == Key.R && humanState == "active") humanReload = true;
        }
        if (humanState == "active" && e is InputEventMouseMotion m)
            humanMouse += m.ScreenRelative;
    }

    void HumanBegin()
    {
        if (humanState != "ready") return;
        humanState = "active"; humanMenu!.Hide(); humanMouse = Vector2.Zero;
        HumanGuideRefresh(true);
        humanJumpReady = false;
        if (!HumanTest) Input.MouseMode = Input.MouseModeEnum.Captured;
        else
        {
            Main.I.ApplyFps(); // Exercise the focus-notification path in native QA.
            GD.Print($"[native-timing] after_focus fps_cap={Engine.MaxFps}");
        }
        startedWall = lastTickWall = Clock; accumulator = 0;
        HumanCapture();
    }

    static int Nearest(float value, float[] bins) => Enumerable.Range(0, bins.Length).MinBy(i => Math.Abs(bins[i] - value));
    int[] HumanKeys()
    {
        var a = config.Noop;
        if (HumanTest) {
            a[0] = 1;
            if (CmdLine.Has("--agent-human-test-pan")) a[3] = Nearest(steps % 80 < 40 ? 4 : -4, config.YawBins);
            return a;
        }
        bool Held(Key key) => Input.IsPhysicalKeyPressed(key);
        a[0] = Held(Key.W) && !Held(Key.S) ? 1 : Held(Key.S) && !Held(Key.W) ? 2 : 0;
        a[1] = Held(Key.A) && !Held(Key.D) ? 1 : Held(Key.D) && !Held(Key.A) ? 2 : 0;
        a[2] = Held(Key.Ctrl) ? 2 : Held(Key.Shift) ? 1 : 0;
        // Match upstream default sensitivity, then quantize to the model's actual action bins.
        a[3] = Nearest(humanMouse.X * .028f, config.YawBins);
        a[4] = Nearest(-humanMouse.Y * .028f, config.PitchBins);
        if (!Held(Key.Space)) humanJumpReady = true;
        a[5] = humanJumpReady && Held(Key.Space) ? 1 : 0;
        a[6] = Input.IsMouseButtonPressed(MouseButton.Left) ? 1 : 0;
        a[7] = humanReload ? 1 : 0;
        humanMouse = Vector2.Zero; humanReload = false;
        return a;
    }

    async void HumanCapture()
    {
        humanCapturing = true;
        double captureStart = Clock;
        try
        {
            HumanTestStall("capture");
            // Simulation is stopped exactly at an action boundary until its pixels have rendered.
            await ToSignal(RenderingServer.Singleton, RenderingServer.SignalName.FramePostDraw);
            if (humanState != "active" || !IsInsideTree()) return;
            double readbackStart = Clock;
            humanRenderWaitMs = (readbackStart - captureStart) * 1000;
            using var image = GetViewport().GetTexture().GetImage();
            if (image == null || image.IsEmpty()) throw new IOException("Empty rendered image");
            image.Resize(config.Width, config.Height, Image.Interpolation.Bilinear);
            image.Convert(Image.Format.Rgb8);
            humanReadbackMs = (Clock - readbackStart) * 1000;
            humanCaptureMs = (Clock - captureStart) * 1000;
            // Initial shader/readback warm-up is loading, not delayed gameplay.
            if (ticks == 0) { startedWall = lastTickWall = Clock; accumulator = 0; }
            else if (!HumanAdvanceClock("capture")) return;
            humanPixels = image.GetData(); humanBefore = Info();
            if (HumanTest && steps == 0 && CmdLine.Has("--agent-human-test-guide-images"))
            {
                image.SavePng(Path.Combine(humanDirectory!, "game-observation.png"));
                using var guideImage = humanGuideWindow?.GetTexture().GetImage();
                guideImage?.SavePng(Path.Combine(humanDirectory!, "teacher-guide.png"));
            }
            humanAction = HumanKeys(); appliedAction = humanAction;
            humanSampleWall = Clock; humanPausedSeconds = 0;
            humanCaptureMs = (humanSampleWall - captureStart) * 1000;
            session.AgentSetAction(config, humanAction);
        }
        catch (Exception e) { GD.PrintErr(e.Message); HumanStop("capture_failed"); }
        finally { humanCapturing = false; }
    }

    void HumanProcess()
    {
        HumanGuideRefresh();
        if (humanState != "active") return;
        if (humanCapturing && ticks == 0) { lastTickWall = Clock; accumulator = 0; return; }
        if (!HumanAdvanceClock("action")) return;
        if (humanCapturing || humanPixels == null) return;
        while (accumulator >= 1.0 / config.PhysicsHz && humanState == "active")
        {
            Tick(); accumulator -= 1.0 / config.PhysicsHz;
            if (ticks % config.ActionRepeat != 0)
            {
                if (HumanTestStall("action") && !HumanAdvanceClock("action")) return;
                continue;
            }
            var after = Info();
            bool written = humanWriter!.Append(humanPixels, new {
                index = steps - 1, action = humanAction, before = humanBefore, after,
                timing = new { clock = "native_fixed_step", alignment = "rendered_pre_action_v1",
                    input_sampled = humanSampleWall, completed = Clock,
                    capture_ms = humanCaptureMs, process_interval_ms = humanLastFrameMs,
                    render_wait_ms = humanRenderWaitMs, readback_ms = humanReadbackMs,
                    pacing = "bounded_pause_v1", paused_seconds = humanPausedSeconds,
                    simulation_dt = config.ActionSeconds }
            });
            humanPixels = null;
            rewardPending = 0; rewards.Clear();
            if (Clock - humanLastTitleWall >= 1)
            {
                humanLastTitleWall = Clock;
                GetWindow().Title = $"本机录制 · 第 {steps} 步 · Esc 放弃" +
                    (humanRecoveryCount > 0 ? $" · 已恢复 {humanRecoveryCount} 次短暂停顿" : " · 到终点自动结束");
            }
            if (!written) { HumanStop("writer_queue_full"); return; }
            if (resetRequired || (HumanTest && steps >= HumanTestSteps))
            {
                humanFinal = Info(); humanState = "review"; Input.MouseMode = Input.MouseModeEnum.Visible;
                humanMenu!.Text = success ? "任务完成！\n按 Enter 确认保存示范，Esc 放弃" : "本次未完成目标\n按 Esc 返回选关，本段不会进入训练";
                if (!success && reason == "timeout") humanMenu.Text += $"\n已用完 {config.MaxSteps * config.ActionSeconds:F0} 秒；请按旁边路线图到绿色终点。";
                humanMenu.Show();
                HumanGuideRefresh(true);
                if (HumanTest) HumanFinish(false);
                return;
            }
            HumanCapture(); return;
        }
    }

    bool HumanAdvanceClock(string phase)
    {
        double now = Clock, elapsed = now - lastTickWall; lastTickWall = now;
        humanLastFrameMs = elapsed * 1000;
        accumulator += elapsed;
        if (accumulator <= .25) return true;
        // The world has NOT advanced during this backlog. Discard it rather than replaying a
        // held key at fast-forward speed. Keep the same recorded action for any partial block.
        double lost = accumulator;
        while (humanRecoveries.Count > 0 && now - humanRecoveries.Peek() > 5) humanRecoveries.Dequeue();
        GD.Print($"[native-timing] phase={phase} step={steps} backlog_ms={lost*1000:F1} process_ms={humanLastFrameMs:F1} capture_ms={humanCaptureMs:F1} render_wait_ms={humanRenderWaitMs:F1} readback_ms={humanReadbackMs:F1} fps_cap={Engine.MaxFps}");
        if (lost > .75 || droppedSeconds + lost > 2 || humanRecoveries.Count >= 2)
        {
            HumanStop("sustained_render_lag"); return false;
        }
        humanRecoveries.Enqueue(now); humanRecoveryCount++;
        droppedSeconds += lost; accumulator = 0;
        // Only time after this action was sampled belongs in its pause accounting; render
        // stalls before sampling are instead visible in before/after dropped_seconds.
        if (humanPixels != null) humanPausedSeconds += Math.Min(lost, Math.Max(0, now - humanSampleWall));
        humanMouse = Vector2.Zero; humanReload = false;
        return true;
    }

    bool HumanTestStall(string phase)
    {
        // Deliberate stalls exercise recovery in real-engine QA only, never in human captures.
        if (!HumanTest || steps < 10 || humanTestStallStep == steps ||
            (CmdLine.After("--agent-human-test-stall-phase") ?? "action") != phase ||
            !int.TryParse(CmdLine.After("--agent-human-test-stall-ms"), out int ms)) return false;
        int every = int.TryParse(CmdLine.After("--agent-human-test-stall-every"), out int value) ? Math.Max(1, value) : config.MaxSteps;
        if ((steps - 10) % every != 0) return false;
        humanTestStallStep = steps;
        System.Threading.Thread.Sleep(Math.Clamp(ms, 1, 2000));
        return true;
    }

    void HumanStop(string issue)
    {
        if (humanState is "saving" or "finished") return;
        GD.Print($"[native-timing] stop={issue} step={steps} backlog_ms={accumulator*1000:F1} process_ms={humanLastFrameMs:F1} capture_ms={humanCaptureMs:F1}");
        humanWriter?.Issues.Add(issue); humanFinal ??= Info(); humanState = "review";
        session.AgentSetAction(config, config.Noop);
        Input.MouseMode = Input.MouseModeEnum.Visible;
        string explanation = issue switch {
            "window_lost_focus" => "切换了窗口；请重新选关再录制。",
            "sustained_render_lag" => $"画面处理持续跟不上：本次积压 {accumulator*1000:F0} 毫秒。\n已自动恢复 {humanRecoveryCount} 次短暂停顿；继续录制会影响示范质量。\n详细耗时已保存在录制日志中，请先关闭其他游戏窗口再重试。",
            "writer_queue_full" => "硬盘保存跟不上，请检查磁盘空间和占用。",
            "capture_failed" => "未能获取游戏画面，请重新启动本关。",
            "user_aborted" => "已放弃本次。",
            _ => "本段数据不完整，请重新录制。"
        };
        if (humanMenu != null) { humanMenu.Text = "录制已停止，本段不进入训练。\n按 Esc 返回选关\n" + explanation; humanMenu.Show(); }
        HumanGuideRefresh(true);
        if (HumanTest) HumanFinish(false);
    }

    void HumanFinish(bool accepted)
    {
        if (humanState is "saving" or "finished") return;
        humanState = "saving";
        try
        {
            humanWriter!.Finish(new { schema = "valorant-native-v1", engine = EngineIdentity, source = HumanTest ? "script_test" : "human",
                completion_rule = config.CompletionRule,
                engine_seed = seed, final = humanFinal ?? Info(), steps = humanWriter.Count,
                performance = new { recovered_stalls = humanRecoveryCount, dropped_seconds = droppedSeconds, fps_cap = Engine.MaxFps },
                accepted = accepted && success && humanWriter.Issues.Count == 0 && !HumanTest,
                complete = resetRequired, quality_issues = humanWriter.Issues.ToArray() });
            humanState = "finished"; GetTree().Quit();
        }
        catch (Exception e) { GD.PrintErr("Capture save failed: " + e.Message); humanState = "finished"; GetTree().Quit(2); }
    }
    void HumanClose()
    {
        if (humanWriter == null || humanState == "finished") return;
        humanWriter.Issues.Add("window_closed");
        humanWriter.Finish(new { schema = "valorant-native-v1", engine = EngineIdentity, source = HumanTest ? "script_test" : "human",
            completion_rule = config.CompletionRule,
            engine_seed = seed, final = humanFinal ?? Info(), steps = humanWriter.Count,
            performance = new { recovered_stalls = humanRecoveryCount, dropped_seconds = droppedSeconds, fps_cap = Engine.MaxFps },
            accepted = false, complete = false, quality_issues = humanWriter.Issues.ToArray() });
        humanState = "finished";
    }
}

sealed class NativeWriter
{
    readonly string directory;
    readonly Channel<(byte[] Pixels, string Row)> queue = Channel.CreateBounded<(byte[], string)>(16);
    readonly Task worker;
    public readonly List<string> Issues = new();
    public int Count { get; private set; }
    public NativeWriter(string path)
    {
        directory = path; Directory.CreateDirectory(path);
        worker = Task.Run(async () => {
            using var rows = new StreamWriter(Path.Combine(directory, "steps.jsonl"));
            int index = 0;
            await foreach (var (pixels, row) in queue.Reader.ReadAllAsync())
            {
                using (var file = File.Create(Path.Combine(directory, $"frame-{index:D6}.rgb.gz")))
                using (var zip = new GZipStream(file, CompressionLevel.Fastest)) zip.Write(pixels);
                await rows.WriteLineAsync(row); index++;
            }
        });
    }
    public bool Append(byte[] pixels, object row)
    {
        if (worker.IsFaulted || !queue.Writer.TryWrite((pixels, JsonSerializer.Serialize(row)))) return false;
        Count++; return true;
    }
    public void Finish(object manifest)
    {
        queue.Writer.TryComplete(); worker.GetAwaiter().GetResult();
        string path = Path.Combine(directory, "native.json");
        File.WriteAllText(path + ".tmp", JsonSerializer.Serialize(manifest));
        File.Move(path + ".tmp", path, true);
    }
}
