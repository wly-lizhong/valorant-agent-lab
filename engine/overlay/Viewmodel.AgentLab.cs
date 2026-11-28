using Godot;

namespace CaValAim.Game.Weapon;

public partial class Viewmodel
{
    /// <summary>Reset presentation without changing weapon ballistics or session RNG.</summary>
    public void AgentReset(int seed)
    {
        SetProcess(false);
        rng = new Random(seed);
        adsIn = speedIn = 0;
        reloadIn = -1;
        crouchIn = false;
        time = equipT = bobPhase = bobAmp = crouchK = adsK = reloadW = 0;
        kPos = kPosV = kRot = kRotV = Vector3.Zero;
        sway = swayV = Vector2.Zero;
        lastYaw = lastPitch = 0;
        haveLook = false;
        boltT = -1;
        reloadBolted = false;
        SetMagHidden(false);
        spareMag.Visible = false;
        gripHand.Transform = gripRest;
        supportHand.Transform = supportRest;
        nextShell = 0;
        foreach (var shell in shells)
        {
            shell.Age = 99;
            shell.Vel = shell.AngVel = Vector3.Zero;
            shell.Node.Visible = false;
            shell.Node.Transform = Transform3D.Identity;
        }
        flash.AgentReset(seed ^ 0x6a09e6);
        _Process(0);
    }

    public void AgentTick(double delta)
    {
        flash._Process(delta);
        _Process(delta);
    }
}
