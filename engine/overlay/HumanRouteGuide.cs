using Godot;

namespace CaValAim.AgentLab;

/// <summary>Teacher-only 2D guidance in a separate native Window/Viewport.</summary>
public partial class HumanRouteGuide : Control
{
    readonly AgentConfig config;
    readonly Font font;
    readonly Vector2[] route;
    readonly Vector2 minimum, maximum;
    Vector3 feet;
    float yaw, goalDistance, seconds;
    int waypoint;
    string state = "ready";
    bool success;
    static readonly Color Ink = new("e8f0f5"), Muted = new("a2b4c3"), Green = new("65e3b6"), Blue = new("6ebeff");

    public HumanRouteGuide(AgentConfig c, Font f)
    {
        config = c; font = f; MouseFilter = MouseFilterEnum.Ignore;
        route = new[] { c.StartPosition }.Concat(c.Waypoints.Select(p => new Vector3(p[0], p[1], p[2])))
            .Append(c.GoalPosition).Select(p => new Vector2(p.X, p.Z)).ToArray();
        minimum = new Vector2(route.Min(p => p.X) - 3, route.Min(p => p.Y) - 3);
        maximum = new Vector2(route.Max(p => p.X) + 3, route.Max(p => p.Y) + 3);
    }

    public void Refresh(Vector3 position, float heading, int next, float remaining, string status, bool completed)
    {
        feet = position; yaw = heading; waypoint = next; seconds = remaining; state = status; success = completed;
        goalDistance = feet.DistanceTo(config.GoalPosition);
        QueueRedraw();
    }

    Vector2 Map(Vector2 p)
    {
        var area = new Vector2(276, 246);
        float scale = Math.Min(area.X / (maximum.X - minimum.X), area.Y / (maximum.Y - minimum.Y));
        return new Vector2(160, 223) + (p - (minimum + maximum) / 2) * scale;
    }
    void Text(string s, float x, float y, Color color, int size = 16) => DrawString(font, new Vector2(x, y), s, fontSize: size, modulate: color);

    public override void _Draw()
    {
        DrawRect(new Rect2(0, 0, 320, 540), new Color("101d28"));
        Text("跟随路线，到绿色终点", 16, 31, Ink, 21);
        Text("终点是 A 入口训练区，并非整个 A Site", 16, 57, Muted, 14);
        Text("路线示意（不显示地图墙体）", 16, 84, Muted, 15);
        DrawRect(new Rect2(14, 96, 292, 256), new Color("172b39"));
        for (int i = 1; i < route.Length; i++) DrawLine(Map(route[i - 1]), Map(route[i]), Blue, 4, true);
        for (int i = 0; i < config.Waypoints.Length - 1; i++)
        {
            var p = Map(route[i + 1]);
            DrawCircle(p, 5, i < waypoint ? Muted : Blue);
            Text((i + 1).ToString(), p.X + 10, p.Y + 5, Muted, 13);
        }
        var start = Map(route[0]);
        DrawCircle(start, 6, Muted); Text("起点", start.X - 40, start.Y + 21, Muted, 14);
        var goal = Map(new Vector2(config.Goal[0], config.Goal[2]));
        float scale = Math.Min(276 / (maximum.X - minimum.X), 246 / (maximum.Y - minimum.Y));
        DrawCircle(goal, Math.Max(9, config.GoalRadius * scale), new Color(0.39f, .89f, .71f, .3f));
        DrawArc(goal, Math.Max(9, config.GoalRadius * scale), 0, Mathf.Tau, 32, Green, 2, true);
        Text("终点", goal.X + 15, goal.Y - 7, Green, 16);
        // Clamp only the drawing, never the world. An explicit message explains off-map positions.
        var raw = Map(new Vector2(feet.X, feet.Z));
        var player = raw.Clamp(new Vector2(23, 106), new Vector2(297, 342));
        var forward = new Vector2(Mathf.Sin(Mathf.DegToRad(yaw)), -Mathf.Cos(Mathf.DegToRad(yaw)));
        var side = new Vector2(-forward.Y, forward.X);
        DrawCircle(player, 9, new Color("101d28"));
        DrawColoredPolygon(new[] { player + forward * 13, player - forward * 8 + side * 7, player - forward * 8 - side * 7 }, Green);
        Text(raw == player ? "绿色箭头是你，尖端是朝向" : "你已走出示意图，请返回蓝色路线", 16, 375, raw == player ? Ink : Green, 15);
        Text($"距终点 {goalDistance:F1} 米   剩余 {Math.Max(0, seconds):F0} 秒", 16, 401, Ink, 17);
        var target = config.Waypoints.Length > 0 && waypoint < config.Waypoints.Length
            ? config.Waypoints[waypoint] : config.Goal;
        var offset = new Vector3(target[0], target[1], target[2]) - feet;
        float angle = Mathf.PosMod(Mathf.RadToDeg(Mathf.Atan2(offset.X, -offset.Z)) - yaw + 180, 360) - 180;
        string direction = Math.Abs(angle) < 15 ? "朝向正确，沿通道前进" : $"向{(angle > 0 ? "右" : "左")}转约 {Math.Abs(angle):F0}°，再前进";
        if (goalDistance <= config.GoalRadius) direction = "已到终点；如有敌人，先停稳处理";
        Text(direction, 16, 429, Green, 16);
        Text(config.CompletionRule == "goal_and_targets" ? "蓝色中途点供参考，不必逐个踩中" : "旧版题目：须按顺序到达每个蓝点", 16, 456, Muted, 15);
        Text("WASD 移动 · 鼠标转向 · Esc 放弃", 16, 482, Muted, 14);
        string message = state == "ready" ? "在游戏窗口按 Enter 开始" : state == "active" ? "正在录制，到达终点自动判定" : success ? "任务完成！在游戏按 Enter 保存" : "本次未完成；在游戏按 Esc 返回";
        Text(message, 16, 519, success ? Green : Ink, 16);
    }
}
