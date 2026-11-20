using Godot;
using CaValAim.Maps;
using CaValAim.Modes;
using CaValAim.World;

namespace CaValAim.AgentLab;

public sealed class NavigationMode : TrainingMode
{
    readonly AgentConfig config;
    readonly MapSpot? spot;
    public NavigationMode(AgentConfig c)
    {
        config = c;
        spot = c.Map == "range" ? null : MapSpots.ByKey(c.Map);
        if (c.RequireRealMap && (spot == null || !RealMap.IsComplete(spot.RealMapFolder)
            || MapAssets.Find(spot.RealMapFolder + "/collision.json") == null))
            throw new InvalidOperationException("Real map assets missing/incomplete. Set maps_dir; blockout fallback is forbidden.");
    }
    public override string Key => "agent_navigation";
    public override string Name => "Navigation";
    public override string Description => "Reach the fixed goal using the first-person image.";
    public override bool Movement => true;
    public override bool Timed => false;
    public override bool SimulatesMatch => true;
    public override bool WeaponSlots => false;
    public override MapSpot? Map => spot;
    public override Vector3 StartFeet => config.StartPosition;
    public override float StartYaw => config.StartYaw;
    public override void Update(float dt) { }

    protected override void Setup()
    {
        if (!config.GoalMarker) return;
        var material = new StandardMaterial3D { AlbedoColor = new Color(0.1f, 0.9f, 0.8f),
            ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded };
        var marker = new MeshInstance3D { Name = "NavigationGoal",
            Mesh = new CylinderMesh { TopRadius = 0.18f, BottomRadius = 0.18f, Height = 1.3f },
            MaterialOverride = material, Position = config.GoalPosition + new Vector3(0, 0.65f, 0) };
        G.World.AddChild(marker);
    }
}
