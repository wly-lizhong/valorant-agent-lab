using Godot;

namespace CaValAim.Game.Fx;

public partial class FxWorld
{
    bool agentVisualClock;
    sealed class AgentParticleTime { public float Age, Pending; }
    readonly Dictionary<GpuParticles3D, AgentParticleTime> agentParticles = new();

    public void AgentReset(int seed)
    {
        SetProcess(false);
        agentVisualClock = true;
        agentParticles.Clear();
        rng = new Random(seed);
        nTracer = nFlash = nHole = nSpark = nDust = nBody = 0;
        foreach (var item in tracers)
        {
            item.Live = false;
            item.Node.Visible = false;
        }
        foreach (var item in flashes) { item.Age = 99; item.Node.Visible = false; }
        foreach (var item in holes) { item.Age = 99; item.Node.Visible = false; }
        int i = 0;
        foreach (var particles in sparks.Concat(dust).Concat(body))
        {
            particles.UseFixedSeed = true;
            particles.Seed = unchecked((uint)(seed + i++));
            particles.SpeedScale = 0;
            particles.Interpolate = false;
            particles.Restart(true);
            particles.Emitting = false;
            particles.Visible = false;
        }
    }

    // Called by the upstream Burst hook; ordinary sessions retain automatic animation.
    void AgentParticleBurst(GpuParticles3D particles)
    {
        if (!agentVisualClock) return;
        particles.Visible = true;
        agentParticles[particles] = new AgentParticleTime();
    }

    public void AgentTick(double delta)
    {
        _Process(delta);
        foreach (var state in agentParticles.Values)
        {
            state.Age += (float)delta;
            state.Pending += (float)delta;
        }
    }

    // Submit once per rendered frame: several simulation ticks can share one draw.
    public void AgentRender()
    {
        foreach (var pair in agentParticles.ToArray())
        {
            if (pair.Value.Pending <= 0) continue;
            pair.Key.RequestParticlesProcess(pair.Value.Pending);
            pair.Value.Pending = 0;
            if (pair.Value.Age > pair.Key.Lifetime + .1)
            {
                pair.Key.Visible = false;
                agentParticles.Remove(pair.Key);
            }
        }
    }
}
