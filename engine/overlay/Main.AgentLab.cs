using Godot;
using CaValAim.AgentLab;
using CaValAim.Audio;
using CaValAim.Core;
using CaValAim.Game;
using CaValAim.UI;
using CaValAim.Valorant;

namespace CaValAim;

public partial class Main
{
    public bool AgentLabActive { get; private set; }
    bool StartAgentLab()
    {
        if (!CmdLine.Has("--agent-config")) return false;
        AgentLabActive = true;
        try
        {
            if (DisplayServer.GetName() == "headless")
                throw new InvalidOperationException("Pixel observations need rendering. Use a GPU display or xvfb-run, not --headless.");
            var config = AgentConfig.Load(CmdLine.After("--agent-config")!);
            Settings = new AppSettings { Quality = 0, Volume = 0, LiveCoach = false, GuideLine = 0,
                UnscoredWarmup = true, CheckUpdates = false, AutoDownloadUpdates = false,
                SupportReminders = false, UseSensOverride = true, SensOverride = 0.4f, UiSounds = false };
            Valorant = new ValorantProfile(); // Training never imports a user's actual game settings.
            Stats = new StatsStore();
            MapOverride = config.Map;
            TierOverride = 0;
            Sfx.Volume = 0;
            AddChild(new Sfx { Name = "Sfx" });
            UiTheme.Init();
            GetTree().Root.Title = "VALORANT Agent Lab";
            GetTree().Root.Size = new Vector2I(config.RenderWidth, config.RenderHeight);
            DisplayServer.WindowSetVsyncMode(DisplayServer.VSyncMode.Disabled);
            Engine.MaxFps = 120;
            var session = new GameSession(() => new NavigationMode(config)) { AgentControlled = true, WarmupSession = true };
            screen = session;
            inGame = true;
            AddChild(session);
            session.AgentReset(config, 0);
            var bridge = new AgentBridge(session, config, int.Parse(CmdLine.After("--agent-port") ?? "0"),
                CmdLine.After("--agent-ready") ?? throw new ArgumentException("Missing --agent-ready"));
            AddChild(bridge);
            GD.Print("[agent-lab] ready; loopback RPC, rendered RGB observations");
        }
        catch (Exception error)
        {
            GD.PrintErr("[agent-lab] startup failed: " + error);
            GetTree().Quit(2);
        }
        return true;
    }
}
