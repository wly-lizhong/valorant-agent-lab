using Godot;

namespace CaValAim.Game.Weapon;

public partial class MuzzleFlash
{
    public void AgentReset(int seed)
    {
        SetProcess(false);
        rng = new Random(seed);
        Hold = false;
        t = 99;
        lastK = -1;
        star.Scale = side.Scale = Vector3.One;
        side.Rotation = Vector3.Zero;
        starMat.SetShaderParameter("rot", 0f);
        starMat.SetShaderParameter("seed", 0f);
        sideMat.SetShaderParameter("seed", 0f);
        _Process(0);
    }
}
