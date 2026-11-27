using Godot;
using CaValAim.Game;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace CaValAim.AgentLab;

public partial class AgentBridge : Node
{
    const string EngineIdentity = "ca-val-aim-v2.15.0-lab-v3";
    record Request(JsonElement Data, TaskCompletionSource<string> Completion);
    readonly GameSession session;
    readonly AgentConfig config;
    readonly TcpListener listener;
    readonly string readyPath;
    readonly bool realtime;
    readonly ConcurrentQueue<Request> requests = new();
    readonly CancellationTokenSource cancel = new();
    bool busy, resetRequired = true, running, terminated, truncated, success;
    int steps, ticks, frameId, seed, lastKills, lastHits, lastShots, lastMovingShots, lastWaypoint, watchdogReleases;
    float previousDistance, bestDistance, rewardPending;
    string reason = "";
    double lastTickWall, lastInputWall, startedWall, accumulator, droppedSeconds;
    int[] appliedAction;
    readonly Dictionary<string, float> rewards = new();
    static double Clock => (double)Stopwatch.GetTimestamp() / Stopwatch.Frequency;
    MovementCombatMode? TaskMode => session.Mode as MovementCombatMode;
    float Distance => TaskMode?.Distance ?? session.PlayerFeet.DistanceTo(config.GoalPosition);

    public AgentBridge(GameSession s, AgentConfig c, int port, string ready, bool real = false, string? human = null)
    {
        session = s; config = c; readyPath = ready; realtime = real; appliedAction = c.Noop;
        humanDirectory = human;
        listener = new TcpListener(IPAddress.Loopback, port);
    }
    public override void _Ready()
    {
        if (humanDirectory != null) { HumanSetup(); return; }
        listener.Start(1);
        File.WriteAllText(readyPath, JsonSerializer.Serialize(new {
            protocol = 1, revision = 2, port = ((IPEndPoint)listener.LocalEndpoint).Port,
            pid = System.Environment.ProcessId, scenario = config.Id, map = config.Map,
            action_sizes = config.ActionSizes, clock = realtime ? "realtime" : "lockstep",
            render_device = RenderingServer.GetVideoAdapterName(),
            render_vendor = RenderingServer.GetVideoAdapterVendor()
        }));
        _ = System.Threading.Tasks.Task.Run(Serve);
    }
    async Task Serve()
    {
        try
        {
            while (!cancel.IsCancellationRequested)
            {
                using var client = await listener.AcceptTcpClientAsync(cancel.Token);
                client.NoDelay = true;
                try
                {
                    using var stream = client.GetStream();
                    using var reader = new StreamReader(stream, Encoding.UTF8);
                    using var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
                    while (!cancel.IsCancellationRequested)
                    {
                        var line = await reader.ReadLineAsync(cancel.Token);
                        if (line == null || line.Length > 65536) break;
                        try
                        {
                            using var parsed = JsonDocument.Parse(line);
                            var completion = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
                            requests.Enqueue(new Request(parsed.RootElement.Clone(), completion));
                            await writer.WriteLineAsync(await completion.Task.WaitAsync(cancel.Token));
                        }
                        catch (JsonException) { await writer.WriteLineAsync("{\"ok\":false,\"error\":\"Invalid JSON\"}"); }
                    }
                }
                catch (Exception error) when (error is IOException or SocketException)
                { GD.PrintErr("[agent-lab] client disconnected: " + error.Message); }
            }
        }
        catch (Exception error) when (error is OperationCanceledException or ObjectDisposedException or SocketException)
        { if (!cancel.IsCancellationRequested) GD.PrintErr("[agent-lab] RPC stopped: " + error.Message); }
    }
    void AddReward(string name, float value)
    {
        rewards[name] = rewards.GetValueOrDefault(name) + value;
        rewardPending += value;
    }
    void ScoreStep()
    {
        steps++;
        float distance = Distance;
        bool fell = session.FallRescues > 0 || session.PlayerFeet.Y < -20;
        success = (TaskMode?.Success ?? distance <= config.GoalRadius) && !session.Player.Dead && !fell;
        terminated = success || session.Player.Dead || fell;
        truncated = !terminated && steps >= config.MaxSteps;
        reason = success ? "success" : session.Player.Dead ? "death" : fell ? "fall" : truncated ? "timeout" : "";
        if (TaskMode is { } task)
        {
            // Only new best progress earns shaping: walking back and forth cannot farm it.
            if (task.Waypoint == lastWaypoint)
                AddReward("progress", .05f * Math.Max(0, bestDistance - distance));
            else AddReward("waypoint", .3f * (task.Waypoint - lastWaypoint));
            bestDistance = task.Waypoint == lastWaypoint ? Math.Min(bestDistance, distance) : distance;
            lastWaypoint = task.Waypoint;
            AddReward("hit", .1f * (task.Hits - lastHits));
            AddReward("kill", 2f * (task.Kills - lastKills));
            AddReward("shot_cost", -.005f * (task.Shots - lastShots));
            AddReward("moving_shot", -.01f * (task.MovingShotCount - lastMovingShots));
            AddReward("time", -.002f);
            lastKills = task.Kills; lastHits = task.Hits; lastShots = task.Shots; lastMovingShots = task.MovingShotCount;
        }
        else AddReward("navigation", .5f * (previousDistance - distance) - .01f);
        if (success) AddReward("completion", 5);
        else if (terminated) AddReward("failure", -2);
        previousDistance = distance;
        resetRequired = terminated || truncated;
        if (resetRequired) { running = false; session.AgentSetAction(config, config.Noop); }
    }
    void Tick()
    {
        session.AgentTick(config);
        ticks++;
        if (ticks % config.ActionRepeat == 0) ScoreStep();
    }
    public override void _Process(double delta)
    {
        if (humanDirectory != null) { HumanProcess(); return; }
        if (realtime && running && !resetRequired)
        {
            double now = Clock;
            double elapsed = now - lastTickWall;
            lastTickWall = now;
            // Report clock loss explicitly; a slow renderer cannot silently pass as realtime.
            droppedSeconds += Math.Max(0, elapsed - .5);
            accumulator += Math.Min(elapsed, .5);
            if (now - lastInputWall > .25 && !appliedAction.SequenceEqual(config.Noop))
            {
                appliedAction = config.Noop;
                session.AgentSetAction(config, appliedAction);
                watchdogReleases++;
            }
            while (accumulator >= 1.0 / config.PhysicsHz && !resetRequired)
            {
                Tick();
                accumulator -= 1.0 / config.PhysicsHz;
            }
        }
        if (busy || !requests.TryDequeue(out var request)) return;
        busy = true;
        Handle(request);
    }
    async void Handle(Request request)
    {
        int id = 0;
        try
        {
            var data = request.Data;
            id = data.GetProperty("id").GetInt32();
            string command = data.GetProperty("command").GetString() ?? "";
            if (command == "close")
            {
                running = false;
                request.Completion.TrySetResult(JsonSerializer.Serialize(new { id, ok = true }));
                GetTree().CreateTimer(.1).Timeout += () => GetTree().Quit();
                return;
            }
            if (command == "reset")
            {
                ResetEpisode(data.TryGetProperty("seed", out var value) ? value.GetInt32() : 0);
            }
            else if (command == "step")
            {
                // A realtime episode may finish while the client is still inferring.
                if (resetRequired && !realtime) throw new InvalidOperationException("Call reset before step / after episode end");
                if (!resetRequired)
                {
                    var action = data.GetProperty("action").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                    session.AgentSetAction(config, action);
                    appliedAction = action;
                    lastInputWall = Clock;
                    if (realtime)
                    {
                        if (!running) { startedWall = lastTickWall = Clock; running = true; }
                        int before = steps;
                        while (!resetRequired && steps == before) await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
                    }
                    else for (int i = 0; i < config.ActionRepeat && !resetRequired; i++) Tick();
                }
                else if (steps == 0) throw new InvalidOperationException("Call reset before step");
            }
            else throw new ArgumentException("Unknown command");
            await ToSignal(RenderingServer.Singleton, RenderingServer.SignalName.FramePostDraw);
            using var image = GetViewport().GetTexture().GetImage();
            if (image == null || image.IsEmpty()) throw new InvalidOperationException("Renderer returned no pixels");
            image.Resize(config.Width, config.Height, Image.Interpolation.Bilinear);
            image.Convert(Image.Format.Rgb8);
            var info = Info();
            request.Completion.TrySetResult(JsonSerializer.Serialize(new {
                id, ok = true, protocol = 1, revision = 2, rgb = Convert.ToBase64String(image.GetData()),
                width = config.Width, height = config.Height, reward = rewardPending, terminated, truncated, info
            }));
            rewardPending = 0; rewards.Clear();
        }
        catch (Exception error)
        {
            resetRequired = true; running = false;
            session.AgentSetAction(config, config.Noop);
            request.Completion.TrySetResult(JsonSerializer.Serialize(new { id, ok = false, error = error.Message }));
        }
        finally { busy = false; }
    }
    void ResetEpisode(int newSeed)
    {
        running = false; seed = newSeed;
        session.AgentReset(config, seed);
        steps = ticks = frameId = lastKills = lastHits = lastShots = lastMovingShots = lastWaypoint = watchdogReleases = 0;
        resetRequired = terminated = truncated = success = false;
        reason = ""; rewardPending = 0; rewards.Clear();
        accumulator = droppedSeconds = 0;
        startedWall = lastTickWall = lastInputWall = Clock;
        appliedAction = config.Noop;
        bestDistance = previousDistance = Distance;
    }
    object Info()
    {
        double wall = Clock - startedWall;
        return new {
                scenario = config.Id, family = config.Family, map = config.Map, mode = config.Mode,
                seed, step = steps, frame_id = frameId++, simulation_time = session.Now,
                is_success = success, termination_reason = reason, distance_to_goal = previousDistance,
                // These privileged diagnostics never enter observations or the BC data loader.
                position = new[] { session.PlayerFeet.X, session.PlayerFeet.Y, session.PlayerFeet.Z },
                yaw = session.View.Yaw, pitch = session.View.Pitch,
                view_yaw = session.View.ViewYaw, view_pitch = session.View.ViewPitch,
                speed = session.Mover.Speed, accurate = session.Mover.Accurate,
                ammo = session.Gun?.Ammo ?? 0, hp = session.Player.Hp, shield = session.Player.Shield,
                reloading = session.Gun?.Reloading ?? false,
                shots = session.Mode.Shots, hits = session.Mode.Hits, kills = session.Mode.Kills,
                moving_shots = TaskMode?.MovingShotCount ?? 0,
                waypoint = TaskMode?.Waypoint ?? 0, enemies_spawned = TaskMode?.Spawned ?? 0,
                reward_terms = new Dictionary<string, float>(rewards), action_applied = appliedAction.ToArray(),
                clock = humanDirectory != null ? "native_fixed_step" : realtime ? "realtime" : "lockstep", wall_seconds = wall,
                realtime_factor = wall > 0 ? session.Now / wall : 0,
                dropped_seconds = droppedSeconds, watchdog_releases = watchdogReleases
            };
    }
    public override void _ExitTree()
    {
        HumanClose();
        cancel.Cancel(); listener.Stop(); Mover.KeyOverride = null;
    }
}
