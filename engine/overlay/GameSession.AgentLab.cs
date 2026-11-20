using Godot;
using CaValAim.AgentLab;
using CaValAim.Valorant;

namespace CaValAim.Game;

public partial class GameSession
{
    public bool AgentControlled { get; init; }
    bool agentTicking;
    int[] agentAction = new[] { 0, 0, 0, 2, 1, 0, 0, 0 };
    public void AgentReset(AgentConfig config, int seed)
    {
        if (!AgentControlled) throw new InvalidOperationException("Not an agent session");
        Rng = new Random(seed);
        Mover = new Mover();
        WireMover();
        Mover.Hold(weaponDef);
        Mover.KeyOverride = AgentKey;
        Player.Reset();
        TeleportPlayer(config.StartPosition);
        if (PlayerFeet.DistanceTo(config.StartPosition) > 1.0f)
            throw new InvalidOperationException("Spawn was relocated by collision. Check scenario coordinates.");
        View.Yaw = config.StartYaw + ((float)Rng.NextDouble() * 2 - 1) * config.YawJitter;
        View.Pitch = View.RecoilPitch = View.RecoilYaw = 0;
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
        agentAction = new[] { 0, 0, 0, 2, 1, 0, 0, 0 };
        if (Gun != null)
        {
            Gun.ResetRecoil();
            Gun.Ammo = Gun.Def.Mag;
            Gun.Reserve = Gun.Def.Reserve;
            Gun.ReloadLeft = Gun.EquipLeft = 0;
            Gun.Scoped = false;
            Gun.SpreadNow = 0;
        }
        hud.Visible = false; // No coach, coordinates, score or goal-distance leaks into RGB.
        UpdateCamera();
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

    public void AgentAdvance(AgentConfig config, int[] action)
    {
        var sizes = new[] { 3, 3, 3, 5, 3, 2, 2, 2 };
        if (action.Length != sizes.Length || action.Where((a, i) => a < 0 || a >= sizes[i]).Any())
            throw new ArgumentException("Action must match MultiDiscrete([3,3,3,5,3,2,2,2])");
        bool oldFire = fireHeld;
        agentAction = action;
        float yaw = new[] { -15f, -5f, 0f, 5f, 15f }[action[3]];
        float pitch = new[] { -5f, 0f, 5f }[action[4]];
        if (action[7] == 1) Gun?.StartReload(false);
        fireHeld = action[6] == 1;
        firePressedEdge = fireHeld && !oldFire;
        try
        {
            agentTicking = true;
            for (int i = 0; i < config.ActionRepeat; i++)
            {
                // Relative look, spread over fixed simulation substeps for continuous motion.
                View.Yaw = Mathf.Wrap(View.Yaw + yaw / config.ActionRepeat, -180f, 180f);
                View.Pitch = Mathf.Clamp(View.Pitch + pitch / config.ActionRepeat, -89, 89);
                _Process(1.0 / config.PhysicsHz);
            }
        }
        finally { agentTicking = false; }
    }
}
