using Godot;
using CaValAim.Core;
using CaValAim.Game;
using CaValAim.Game.Bots;
using CaValAim.Maps;
using CaValAim.Modes;
using CaValAim.World;

namespace CaValAim.AgentLab;

/// <summary>One episode, no upstream auto-respawn, previews, guides or omniscient focus.</summary>
public sealed class MovementCombatMode : MovementMode
{
    readonly AgentConfig config;
    readonly MapSpot spot;
    readonly List<(BotCharacter Body, BotBrain Brain)> bots = new();
    public int Spawned { get; private set; }
    public int MovingShotCount => MovingShots;
    public int Waypoint { get; private set; }
    public bool RouteComplete => Waypoint >= config.Waypoints.Length;
    public bool Success => RouteComplete && Kills == Spawned && Distance <= config.GoalRadius;
    public float Distance => ((GameSession)G).PlayerFeet.DistanceTo(Target);
    public Vector3 Target => Point(config.Waypoints[Math.Min(Waypoint, config.Waypoints.Length - 1)]);
    static Vector3 Point(float[] p) => new(p[0], p[1], p[2]);
    public MovementCombatMode(AgentConfig c)
    {
        config = c;
        spot = MapSpots.ByKey(c.Map);
        if (c.RequireRealMap && (!RealMap.IsComplete(spot.RealMapFolder)
            || MapAssets.Find(spot.RealMapFolder + "/collision.json") == null))
            throw new InvalidOperationException("Real map assets are required for movement/combat");
    }
    public override string Key => "agent_move_combat";
    public override string Name => "Movement and combat";
    public override string Description => config.Task;
    public override bool Timed => false;
    public override bool SimulatesMatch => true;
    public override bool WeaponSlots => false;
    public override MapSpot? Map => spot;
    public override Vector3 StartFeet => config.StartPosition;
    public override float StartYaw => config.StartYaw;

    // Setup deliberately does not start a round. AgentReset owns every new episode.
    public void ResetEpisode()
    {
        foreach (var b in bots) if (GodotObject.IsInstanceValid(b.Body)) b.Body.Free();
        bots.Clear();
        Shots = Hits = Kills = HeadHits = Deaths = Score = MovingShots = Spawned = Waypoint = 0;
        KillTimes.Clear();
        KillBanner = (0, 0);
        LastShotZone = HitZone.None;
        foreach (var p in config.Enemies)
        {
            if (Rng.NextDouble() >= config.EnemyProbability) continue;
            var feet = MapMode.StandOnFloor(Point(p), G.Solid);
            if (feet.DistanceTo(Point(p)) > 1f)
                throw new InvalidOperationException("Enemy spawn relocated by collision; fix the scenario");
            // Deterministic appearance; do not use SpawnBot's Random.Shared variant choice.
            var body = BotCharacter.Create(G.Enemy, Main.I.Settings.Outlines, Rng.Next(BotAssets.Variants.Length));
            body.SpawnTime = G.Now;
            G.World.AddChild(body);
            body.Feet = feet;
            body.SetProcess(false); // animation/hitboxes must follow the simulation clock too
            var held = (G.View.Eye - body.Head).Normalized();
            body.FacingYaw = Mathf.RadToDeg(Mathf.Atan2(held.X, -held.Z));
            body._Process(1.0 / config.PhysicsHz);
            var brain = new BotBrain(body, Difficulty.Get(config.EnemyTier).Bot, Rng)
            { HeldDir = held, Enabled = config.EnemyFire };
            bots.Add((body, brain));
            Spawned++;
        }
    }
    public override void Update(float dt)
    {
        foreach (var b in bots)
        {
            if (!GodotObject.IsInstanceValid(b.Body) || b.Body.IsQueuedForDeletion()) continue;
            b.Brain.Update(G, dt);
            b.Body._Process(dt);
        }
        if (!G.Player.Dead && !RouteComplete && Distance <= config.GoalRadius) Waypoint++;
    }
    public override void OnPlayerDied() => Deaths++;
    public override float OnBullet(Vector3 o, Vector3 d, bool accurate)
    {
        CountShot(accurate);
        var hit = ShootBots(bots.Select(b => b.Body), o, d, out _, out _, out float dist, Wall(o, d));
        return hit == null ? float.PositiveInfinity : dist;
    }
}
