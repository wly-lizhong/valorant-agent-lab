using Godot;
using CaValAim.AgentLab;
using CaValAim.Valorant;

namespace CaValAim.Game;

public partial class GameSession
{
    public bool AgentControlled { get; init; }
    bool agentTicking;
    HashSet<ulong>? agentStaticNodes;
    PlayerHud? agentHud;
    int[] agentAction = new[] { 0, 0, 0, 2, 1, 0, 0, 0 };
    public void AgentReset(AgentConfig config, int seed)
    {
        if (!AgentControlled) throw new InvalidOperationException("Not an agent session");
        agentStaticNodes ??= World.GetChildren().Select(n => n.GetInstanceId()).ToHashSet();
        foreach (var node in World.GetChildren())
            if (!agentStaticNodes.Contains(node.GetInstanceId())) node.Free();
        Rng = new Random(seed);
        Mover = new Mover();
        WireMover();
        Mover.Hold(weaponDef);
        Mover.KeyOverride = AgentKey;
        Player.Reset();
        var start = config.StartPosition + new Vector3(((float)Rng.NextDouble() * 2 - 1) * config.StartJitter,
            0, ((float)Rng.NextDouble() * 2 - 1) * config.StartJitter);
        TeleportPlayer(start);
        if (PlayerFeet.DistanceTo(start) > 1.0f)
            throw new InvalidOperationException("Spawn was relocated by collision. Check scenario coordinates.");
        View.Yaw = config.StartYaw + ((float)Rng.NextDouble() * 2 - 1) * config.YawJitter;
        View.Pitch = View.RecoilPitch = View.RecoilYaw = 0;
        View.Pitch = config.StartPitch;
        View.Zoom = View.VisualZoom = 1;
        Now = 0;
        FallRescues = 0;
        lastSafeFeet = config.StartPosition;
        safeCheckT = 0;
        BannerTime = 0;
        wasReloading = false;
        AdsBlend = 0;
        Blind.Reset();
        Countdown = 0;
        State = St.Running;
        CoachSuspended = true;
        ResetTransientInput();
        agentAction = config.Noop;
        if (Gun != null)
        {
            Gun.ResetRecoil();
            Gun.Ammo = config.InitialAmmo;
            Gun.Reserve = Gun.Def.Reserve;
            Gun.ReloadLeft = Gun.EquipLeft = 0;
            Gun.Scoped = false;
            Gun.SpreadNow = 0;
        }
        hud.Visible = false; // No coach, coordinates, score or goal-distance leaks into RGB.
        if (config.PlayerHud && agentHud == null)
        {
            agentHud = new PlayerHud(this);
            hud.GetParent().AddChild(agentHud);
        }
        if (agentHud != null) { agentHud.Visible = config.PlayerHud; agentHud.QueueRedraw(); }
        if (Mode is MovementCombatMode task) task.ResetEpisode();
        UpdateCamera();
        // Visual state belongs to this episode too. Do not consume gameplay RNG.
        viewmodel?.AgentReset(seed ^ 0x51f15e);
        foreach (var fx in World.GetChildren().OfType<CaValAim.Game.Fx.FxWorld>())
            fx.AgentReset(seed ^ 0x137a21);
        UpdateAdsVisuals();
        crosshair.Profile = TrialCrosshair ?? CrosshairCode.Effective;
        crosshair.MovementErrorPx = crosshair.FiringErrorPx = 0;
        crosshair.MinErrorPx = weaponDef == null ? 0 : DegToPx(weaponDef.FirstShotHip);
        crosshair.QueueRedraw();
    }

    bool AgentKey(InputBinding key)
    {
        var p = Main.I.Valorant;
        return (key == p.KeyForward && agentAction[0] == 1)
            || (key == p.KeyBack && agentAction[0] == 2)
            || (key == p.KeyLeft && agentAction[1] == 1)
            || (key == p.KeyRight && agentAction[1] == 2)
            || (key == p.KeyWalk && agentAction[2] == 1)
            || (key == p.KeyCrouch && agentAction[2] == 2)
            || (key == p.KeyJump && agentAction[5] == 1);
    }

    public void AgentSetAction(AgentConfig config, int[] action)
    {
        var sizes = config.ActionSizes;
        if (action.Length != sizes.Length || action.Where((a, i) => a < 0 || a >= sizes[i]).Any())
            throw new ArgumentException("Action must match the scenario's MultiDiscrete action sizes");
        bool oldFire = fireHeld;
        agentAction = action;
        if (action[7] == 1) Gun?.StartReload(false);
        fireHeld = action[6] == 1;
        firePressedEdge = fireHeld && !oldFire;
    }

    public void AgentTick(AgentConfig config)
    {
        try
        {
            agentTicking = true;
            // Relative look, spread over fixed simulation substeps for continuous motion.
            View.Yaw = Mathf.Wrap(View.Yaw + config.YawBins[agentAction[3]] / config.ActionRepeat, -180f, 180f);
            View.Pitch = Mathf.Clamp(View.Pitch + config.PitchBins[agentAction[4]] / config.ActionRepeat, -89, 89);
            _Process(1.0 / config.PhysicsHz);
            viewmodel?.AgentTick(1.0 / config.PhysicsHz);
            foreach (var fx in World.GetChildren().OfType<CaValAim.Game.Fx.FxWorld>())
                fx.AgentTick(1.0 / config.PhysicsHz);
            agentHud?.QueueRedraw();
        }
        finally { agentTicking = false; }
    }

    public void AgentRender()
    {
        foreach (var fx in World.GetChildren().OfType<CaValAim.Game.Fx.FxWorld>())
            fx.AgentRender();
    }
}
