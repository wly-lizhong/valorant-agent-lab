using Godot;
using CaValAim.Game;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace CaValAim.AgentLab;

public partial class AgentBridge : Node
{
    record Request(JsonElement Data, TaskCompletionSource<string> Completion);
    readonly GameSession session;
    readonly AgentConfig config;
    readonly TcpListener listener;
    readonly string readyPath;
    readonly ConcurrentQueue<Request> requests = new();
    readonly CancellationTokenSource cancel = new();
    bool busy, resetRequired = true;
    int steps;
    float previousDistance;
    public AgentBridge(GameSession s, AgentConfig c, int port, string ready)
    {
        session = s; config = c; readyPath = ready;
        listener = new TcpListener(IPAddress.Loopback, port);
    }
    public override void _Ready()
    {
        listener.Start(1);
        File.WriteAllText(readyPath, JsonSerializer.Serialize(new {
            protocol = 1, port = ((IPEndPoint)listener.LocalEndpoint).Port,
            pid = System.Environment.ProcessId, scenario = config.Id, map = config.Map
        }));
        _ = Task.Run(Serve);
    }
    async Task Serve()
    {
        try
        {
            while (!cancel.IsCancellationRequested)
            {
                using var client = await listener.AcceptTcpClientAsync(cancel.Token);
                client.NoDelay = true;
                using var stream = client.GetStream();
                using var reader = new StreamReader(stream, Encoding.UTF8);
                using var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
                while (!cancel.IsCancellationRequested)
                {
                    var line = await reader.ReadLineAsync(cancel.Token);
                    if (line == null) break;
                    if (line.Length > 65536) break;
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
        }
        catch (Exception error) when (error is OperationCanceledException or ObjectDisposedException or SocketException or IOException)
        { if (!cancel.IsCancellationRequested) GD.PrintErr("[agent-lab] RPC stopped: " + error.Message); }
    }
    public override void _Process(double delta)
    {
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
                request.Completion.TrySetResult(JsonSerializer.Serialize(new { id, ok = true }));
                GetTree().CreateTimer(0.1).Timeout += () => GetTree().Quit();
                return;
            }
            float reward = 0;
            bool success = false, terminated = false, truncated = false;
            if (command == "reset")
            {
                int seed = data.TryGetProperty("seed", out var value) ? value.GetInt32() : 0;
                session.AgentReset(config, seed);
                steps = 0;
                resetRequired = false;
                previousDistance = session.PlayerFeet.DistanceTo(config.GoalPosition);
            }
            else if (command == "step")
            {
                if (resetRequired) throw new InvalidOperationException("Call reset before step / after episode end");
                var action = data.GetProperty("action").EnumerateArray().Select(x => x.GetInt32()).ToArray();
                session.AgentAdvance(config, action);
                steps++;
                float distance = session.PlayerFeet.DistanceTo(config.GoalPosition);
                bool fell = session.FallRescues > 0 || session.PlayerFeet.Y < -20;
                success = distance <= config.GoalRadius && !session.Player.Dead && !fell;
                terminated = success || session.Player.Dead || fell;
                truncated = !terminated && steps >= config.MaxSteps;
                reward = 0.5f * (previousDistance - distance) - 0.01f;
                if (success) reward += 5;
                else if (terminated) reward -= 1;
                previousDistance = distance;
                resetRequired = terminated || truncated;
            }
            else throw new ArgumentException("Unknown command");
            // Read pixels only after the frame reflecting this action has actually been rendered.
            await ToSignal(RenderingServer.Singleton, RenderingServer.SignalName.FramePostDraw);
            using var image = GetViewport().GetTexture().GetImage();
            if (image == null || image.IsEmpty()) throw new InvalidOperationException("Renderer returned no pixels");
            image.Resize(config.Width, config.Height, Image.Interpolation.Bilinear);
            image.Convert(Image.Format.Rgb8);
            string rgb = Convert.ToBase64String(image.GetData());
            request.Completion.TrySetResult(JsonSerializer.Serialize(new {
                id, ok = true, protocol = 1, rgb, width = config.Width, height = config.Height,
                reward, terminated, truncated,
                info = new { scenario = config.Id, map = config.Map, step = steps,
                    simulation_time = session.Now, is_success = success, distance_to_goal = previousDistance,
                    // Diagnostics only: Python never includes these privileged fields in policy observations.
                    position = new[] { session.PlayerFeet.X, session.PlayerFeet.Y, session.PlayerFeet.Z },
                    ammo = session.Gun?.Ammo ?? 0, hp = session.Player.Hp }
            }));
        }
        catch (Exception error)
        {
            resetRequired = true;
            request.Completion.TrySetResult(JsonSerializer.Serialize(new { id, ok = false, error = error.Message }));
        }
        finally { busy = false; }
    }
    public override void _ExitTree()
    {
        cancel.Cancel();
        listener.Stop();
        Mover.KeyOverride = null;
    }
}
