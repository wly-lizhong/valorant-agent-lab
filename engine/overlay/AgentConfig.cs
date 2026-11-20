using Godot;
using System.Text.Json;

namespace CaValAim.AgentLab;

public sealed class AgentConfig
{
    public string Id { get; set; } = "navigation";
    public string Map { get; set; } = "lotus";
    public float[] Start { get; set; } = new float[] { -18, 0, 30.5f };
    public float StartYaw { get; set; } = 10.6f;
    public float[] Goal { get; set; } = new float[] { -17.26f, 0, 26.57f };
    public float GoalRadius { get; set; } = 0.8f;
    public float YawJitter { get; set; } = 5;
    public int MaxSteps { get; set; } = 300;
    public int PhysicsHz { get; set; } = 60;
    public int ActionRepeat { get; set; } = 6;
    public int Width { get; set; } = 160;
    public int Height { get; set; } = 90;
    public int RenderWidth { get; set; } = 640;
    public int RenderHeight { get; set; } = 360;
    public bool GoalMarker { get; set; } = true;
    public bool RequireRealMap { get; set; } = true;
    public static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    public Vector3 StartPosition => new(Start[0], Start[1], Start[2]);
    public Vector3 GoalPosition => new(Goal[0], Goal[1], Goal[2]);

    public static AgentConfig Load(string path)
    {
        var c = JsonSerializer.Deserialize<AgentConfig>(File.ReadAllText(path), Json)
            ?? throw new ArgumentException("Empty scenario");
        if (c.Start.Length != 3 || c.Goal.Length != 3 || !c.Start.Concat(c.Goal).All(float.IsFinite)
            || !float.IsFinite(c.StartYaw) || !float.IsFinite(c.YawJitter) || c.YawJitter < 0 || c.YawJitter > 180
            || !float.IsFinite(c.GoalRadius) || c.GoalRadius <= 0 || c.GoalRadius > 10
            || c.MaxSteps < 1 || c.MaxSteps > 100000 || c.PhysicsHz != 60
            || c.ActionRepeat < 1 || c.ActionRepeat > 30 || c.Width < 64 || c.Height < 64
            || c.Width > 1024 || c.Height > 1024 || c.RenderWidth < c.Width || c.RenderHeight < c.Height
            || c.RenderWidth > 1920 || c.RenderHeight > 1080)
            throw new ArgumentException("Invalid scenario dimensions, timing or coordinates");
        if (c.Map == "range" && c.RequireRealMap)
            throw new ArgumentException("Range cannot satisfy require_real_map");
        if (c.Map != "range" && !Maps.MapSpots.All.Any(s => s.Key == c.Map))
            throw new ArgumentException("Unknown map key");
        return c;
    }
}
