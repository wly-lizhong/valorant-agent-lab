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
    bool HumanTest => CmdLine.Has("--agent-human-test");

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
        humanMenu.AddThemeFontOverride("font", new SystemFont { FontNames = new[] { "Microsoft YaHei", "Noto Sans CJK SC", "sans-serif" } });
        humanMenu.AddThemeFontSizeOverride("font_size", 22);
        humanMenu.AddThemeColorOverride("font_shadow_color", Colors.Black);
        humanMenu.AddThemeConstantOverride("shadow_offset_x", 2);
        humanMenu.AddThemeConstantOverride("shadow_offset_y", 2);
        var layer = new CanvasLayer(); AddChild(layer); layer.AddChild(humanMenu);
        humanMenu.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        Input.MouseMode = Input.MouseModeEnum.Visible;
        GetWindow().Title = "本机录制 · 按 Enter 开始";
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
            humanMouse += m.Relative;
    }

    void HumanBegin()
    {
        if (humanState != "ready") return;
        humanState = "active"; humanMenu!.Hide(); humanMouse = Vector2.Zero;
        humanJumpReady = false;
        if (!HumanTest) Input.MouseMode = Input.MouseModeEnum.Captured;
        startedWall = lastTickWall = Clock; accumulator = 0;
        HumanCapture();
    }

    static int Nearest(float value, float[] bins) => Enumerable.Range(0, bins.Length).MinBy(i => Math.Abs(bins[i] - value));
    int[] HumanKeys()
    {
        var a = config.Noop;
        if (HumanTest) { a[0] = 1; return a; }
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
        try
        {
            // Simulation is stopped exactly at an action boundary until its pixels have rendered.
            await ToSignal(RenderingServer.Singleton, RenderingServer.SignalName.FramePostDraw);
            if (humanState != "active" || !IsInsideTree()) return;
            using var image = GetViewport().GetTexture().GetImage();
            if (image == null || image.IsEmpty()) throw new IOException("Empty rendered image");
            image.Resize(config.Width, config.Height, Image.Interpolation.Bilinear);
            image.Convert(Image.Format.Rgb8);
            // Initial shader/readback warm-up is loading, not delayed gameplay.
            if (ticks == 0) { startedWall = lastTickWall = Clock; accumulator = 0; }
            humanPixels = image.GetData(); humanBefore = Info();
            humanAction = HumanKeys(); appliedAction = humanAction;
            humanSampleWall = Clock;
            session.AgentSetAction(config, humanAction);
        }
        catch (Exception e) { GD.PrintErr(e.Message); HumanStop("capture_failed"); }
        finally { humanCapturing = false; }
    }

    void HumanProcess()
    {
        if (humanState != "active") return;
        if (humanCapturing && ticks == 0) { lastTickWall = Clock; accumulator = 0; return; }
        double now = Clock, elapsed = now - lastTickWall; lastTickWall = now;
        accumulator += elapsed;
        // Do not fast-forward accumulated movement after a long stall.
        if (accumulator > .25) { HumanStop("render_or_disk_too_slow"); return; }
        if (humanCapturing || humanPixels == null) return;
        while (accumulator >= 1.0 / config.PhysicsHz && humanState == "active")
        {
            Tick(); accumulator -= 1.0 / config.PhysicsHz;
            if (ticks % config.ActionRepeat != 0) continue;
            var after = Info();
            bool written = humanWriter!.Append(humanPixels, new {
                index = steps - 1, action = humanAction, before = humanBefore, after,
                timing = new { clock = "native_fixed_step", alignment = "rendered_pre_action_v1",
                    input_sampled = humanSampleWall, completed = Clock,
                    simulation_dt = config.ActionSeconds }
            });
            humanPixels = null;
            rewardPending = 0; rewards.Clear();
            GetWindow().Title = $"本机录制 · 第 {steps} 步 · Esc 放弃 · 到终点自动结束";
            if (!written) { HumanStop("writer_queue_full"); return; }
            if (resetRequired || (HumanTest && steps >= 20))
            {
                humanState = "review"; Input.MouseMode = Input.MouseModeEnum.Visible;
                humanMenu!.Text = success ? "任务完成！\n按 Enter 确认保存示范，Esc 放弃" : "本次未完成目标\n按 Esc 返回选关，本段不会进入训练";
                humanMenu.Show();
                if (HumanTest) HumanFinish(false);
                return;
            }
            HumanCapture(); return;
        }
    }

    void HumanStop(string issue)
    {
        if (humanState is "saving" or "finished") return;
        humanWriter?.Issues.Add(issue); humanState = "review";
        session.AgentSetAction(config, config.Noop);
        Input.MouseMode = Input.MouseModeEnum.Visible;
        string explanation = issue switch {
            "window_lost_focus" => "切换了窗口；请重新选关再录制。",
            "render_or_disk_too_slow" => "本机出现明显卡顿；请关闭占用较高的程序后重试。",
            "writer_queue_full" => "硬盘保存跟不上，请检查磁盘空间和占用。",
            "capture_failed" => "未能获取游戏画面，请重新启动本关。",
            "user_aborted" => "已放弃本次。",
            _ => "本段数据不完整，请重新录制。"
        };
        if (humanMenu != null) { humanMenu.Text = "录制已停止，本段不进入训练。\n按 Esc 返回选关\n" + explanation; humanMenu.Show(); }
        if (HumanTest) HumanFinish(false);
    }

    void HumanFinish(bool accepted)
    {
        if (humanState is "saving" or "finished") return;
        humanState = "saving";
        try
        {
            humanWriter!.Finish(new { schema = "valorant-native-v1", engine = EngineIdentity, source = HumanTest ? "script_test" : "human",
                engine_seed = seed, final = Info(), steps = humanWriter.Count,
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
            engine_seed = seed, final = Info(), steps = humanWriter.Count,
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
