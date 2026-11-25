using Godot;
using CaValAim.Game;

namespace CaValAim.AgentLab;

/// <summary>Only information a player can see. No target, distance, bot count or reward.</summary>
public partial class PlayerHud : Control
{
    readonly GameSession session;
    public PlayerHud(GameSession s) { session = s; MouseFilter = MouseFilterEnum.Ignore; }
    public override void _Draw()
    {
        var size = GetViewportRect().Size;
        float scale = size.X / 640f;
        int fontSize = Math.Max(18, (int)(20 * scale));
        DrawRect(new Rect2(0, size.Y - 40 * scale, size.X, 40 * scale), new Color(0, 0, 0, 0.65f));
        DrawString(ThemeDB.FallbackFont, new Vector2(12 * scale, size.Y - 12 * scale),
            $"HP {Math.Max(0, (int)session.Player.Hp)}   SHIELD {Math.Max(0, (int)session.Player.Shield)}",
            fontSize: fontSize, modulate: Colors.White);
        string ammo = session.Gun == null ? "" : $"{session.Gun.Ammo} / {session.Gun.Reserve}";
        if (session.Gun?.Reloading == true) ammo = "RELOADING  " + ammo;
        DrawString(ThemeDB.FallbackFont, new Vector2(size.X - 240 * scale, size.Y - 12 * scale),
            ammo, fontSize: fontSize, modulate: Colors.White);
    }
}
