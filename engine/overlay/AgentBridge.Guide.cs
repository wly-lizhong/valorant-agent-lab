using Godot;
using CaValAim.Core;

namespace CaValAim.AgentLab;

public partial class AgentBridge
{
    Window? humanGuideWindow;
    HumanRouteGuide? humanGuide;
    double humanGuideUpdated;

    void HumanGuideSetup(Font font)
    {
        if (CmdLine.Has("--agent-human-test-no-guide")) return;
        var game = GetWindow();
        var usable = DisplayServer.ScreenGetUsableRect(game.CurrentScreen);
        int totalWidth = game.Size.X + 332;
        var origin = usable.Position + new Vector2I(Math.Max(8, (usable.Size.X - totalWidth) / 2), Math.Max(38, (usable.Size.Y - 540) / 2));
        game.Position = origin;
        humanGuideWindow = new Window {
            Visible = false, ForceNative = true, Title = "路线与终点 · 操作请留在游戏窗口",
            Unfocusable = true, Transient = true, Unresizable = true, Disable3D = true,
            Size = new Vector2I(320, 540), Position = new Vector2I(
                Math.Min(origin.X + game.Size.X + 12, usable.End.X - 328), origin.Y)
        };
        AddChild(humanGuideWindow);
        humanGuide = new HumanRouteGuide(config, font);
        humanGuideWindow.AddChild(humanGuide);
        humanGuide.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        humanGuideWindow.CloseRequested += () => { }; // Closing the guide must not silently remove the only destination cue.
        humanGuideWindow.Show();
        if (humanGuideWindow.IsEmbedded()) throw new InvalidOperationException("Teacher guide must have a separate native viewport");
        DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Disabled, humanGuideWindow.GetWindowId());
        GD.Print("[native-guide] separate_viewport=true game_viewport=" + GetViewport().GetInstanceId()
            + " guide_viewport=" + humanGuide.GetViewport().GetInstanceId());
        HumanGuideRefresh(true);
    }

    void HumanGuideRefresh(bool force = false)
    {
        if (humanGuide == null || (!force && Clock - humanGuideUpdated < .2)) return;
        humanGuideUpdated = Clock;
        humanGuide.Refresh(session.PlayerFeet, session.View.Yaw, TaskMode?.Waypoint ?? 0,
            (config.MaxSteps - steps) * config.ActionSeconds, humanState, success);
        // A static 2D guide should not repaint at the game's 60 FPS. Retain its texture
        // between the 5 Hz position updates, including when no game simulation is running.
        RenderingServer.ViewportSetUpdateMode(humanGuideWindow!.GetViewportRid(), RenderingServer.ViewportUpdateMode.Once);
    }
}
