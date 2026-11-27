using Godot;
using System.Text.Json;

namespace CaValAim.AgentLab;

public sealed class AgentConfig
{
    public string Id { get; set; } = "navigation";
    public string Map { get; set; } = "lotus";
    public string Mode { get; set; } = "navigation";
    public string Family { get; set; } = "navigation-selftest";
    public string Task { get; set; } = "Reach the fixed destination.";
    public string CompletionRule { get; set; } = "ordered_waypoints";
    public float[] Start { get; set; } = new float[] { -18, 0, 30.5f };
    public float StartYaw { get; set; } = 10.6f;
    public float[] Goal { get; set; } = new float[] { -17.26f, 0, 26.57f };
    public float GoalRadius { get; set; } = 0.8f;
    public float YawJitter { get; set; } = 5;
    public float StartPitch { get; set; }
    public float StartJitter { get; set; }
    public float[][] Waypoints { get; set; } = Array.Empty<float[]>();
    // Teacher window only. Never changes engine progress, rewards, success or policy RGB.
    public int TeacherWaypointStart { get; set; }
    public float[][] Enemies { get; set; } = Array.Empty<float[]>();
    public float EnemyProbability { get; set; } = 1;
    public bool EnemyFire { get; set; } = true;
    public int EnemyTier { get; set; }
    public int InitialAmmo { get; set; } = 25;
    public bool PlayerHud { get; set; }
    public float[] YawBins { get; set; } = new[] { -15f, -5f, 0f, 5f, 15f };
    public float[] PitchBins { get; set; } = new[] { -5f, 0f, 5f };
    public int[] ActionSizes => new[] { 3, 3, 3, YawBins.Length, PitchBins.Length, 2, 2, 2 };
    public int[] Noop => new[] { 0, 0, 0, Array.IndexOf(YawBins, 0f), Array.IndexOf(PitchBins, 0f), 0, 0, 0 };
    public float ActionSeconds => (float)ActionRepeat / PhysicsHz;
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
        bool GoodPoint(float[] p) => p.Length == 3 && p.All(float.IsFinite);
        bool GoodBins(float[] b) => b.Length >= 3 && b.Length <= 31 && b.Count(x => x == 0) == 1
            && b.All(x => float.IsFinite(x) && Math.Abs(x) <= 45)
            && b.Zip(b.Skip(1), (a, z) => a < z).All(x => x);
        if (c.Mode is not ("navigation" or "move_combat")
            || c.CompletionRule is not ("ordered_waypoints" or "goal_and_targets")
            || !GoodBins(c.YawBins) || !GoodBins(c.PitchBins)
            || !float.IsFinite(c.StartPitch) || Math.Abs(c.StartPitch) > 85
            || !float.IsFinite(c.StartJitter) || c.StartJitter < 0 || c.StartJitter > 2
            || c.Waypoints.Length > 32 || c.Enemies.Length > 8
            || c.TeacherWaypointStart < 0 || c.TeacherWaypointStart > c.Waypoints.Length
            || (c.CompletionRule == "ordered_waypoints" && c.TeacherWaypointStart != 0)
            || !c.Waypoints.All(GoodPoint) || !c.Enemies.All(GoodPoint)
            || !float.IsFinite(c.EnemyProbability) || c.EnemyProbability < 0 || c.EnemyProbability > 1
            || c.EnemyTier < 0 || c.EnemyTier > 4 || c.InitialAmmo < 0 || c.InitialAmmo > 25)
            throw new ArgumentException("Invalid task, action bins, enemy or reset configuration");
        if (c.Mode == "move_combat" && (c.GoalMarker || !c.PlayerHud || c.Waypoints.Length == 0))
            throw new ArgumentException("move_combat requires unmarked waypoints and the player HUD");
        return c;
    }
}
