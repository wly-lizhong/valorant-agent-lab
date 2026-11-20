"""Prepare pinned upstream source + overlay and project-local Godot/.NET tools."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import zipfile
from download import checksum, fetch

ROOT = Path(__file__).resolve().parents[1]


def unpack_zip(archive: Path, destination: Path, strip_root: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            parts = Path(entry.filename).parts
            if strip_root:
                parts = parts[1:]
            if not parts:
                continue
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes destination")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def patch_once(path: Path, original: str, replacement: str) -> None:
    content = path.read_text(encoding="utf-8-sig")
    if replacement in content:
        return
    if content.count(original) != 1:
        raise ValueError(f"Upstream patch anchor mismatch: {path.name}")
    path.write_text(content.replace(original, replacement), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--maps-dir", type=Path, help="Directory containing maps/<map>/")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--platform", choices=["linux", "windows"])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--source-only", action="store_true", help="Prepare and patch source without downloading tools")
    args = parser.parse_args()
    target_os = args.platform or ("windows" if os.name == "nt" else "linux")
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    upstream = lock["upstream"]
    source_zip = args.source_archive or cache / "upstream-v2.11.0.zip"
    if args.source_archive:
        if checksum(source_zip) != upstream["sha256"]:
            raise ValueError("Local source archive differs from pinned archive")
    else:
        fetch(upstream["url"], source_zip, upstream["sha256"])
    godot_spec = lock["godot_" + target_os]
    godot_zip = cache / f"godot-{target_os}.zip"
    sdk_tar = None
    if not args.source_only:
        fetch(godot_spec["url"], godot_zip, godot_spec["sha256"])
    if target_os == "linux" and not args.source_only:
        sdk = lock["dotnet_linux"]
        sdk_tar = fetch(sdk["url"], cache / "dotnet-linux.tar.gz", sdk["sha512"], "sha512")
    if args.download_only:
        print("Verified downloads ready in .cache", flush=True)
        return
    if (target_os == "windows") != (os.name == "nt"):
        raise ValueError("Run extraction/build on the target OS (download-only supports cross-platform)")
    upstream_dir = ROOT / "external" / "CA-VAL-AIM"
    stamp = upstream_dir / ".agent-lab-source.json"
    if not stamp.exists():
        if upstream_dir.exists() and any(upstream_dir.iterdir()):
            raise ValueError("Unrecognized upstream directory; preserving it")
        unpack_zip(source_zip, upstream_dir, strip_root=True)
        stamp.write_text(json.dumps(upstream), encoding="utf-8")
    elif json.loads(stamp.read_text())["sha256"] != upstream["sha256"]:
        raise ValueError("Upstream stamp differs from lock")
    project = upstream_dir / "CaValAim"
    for overlay in (ROOT / "engine" / "overlay").glob("*.cs"):
        destination = project / "src" / "AgentLab" / overlay.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay, destination)
    main_file = project / "src" / "Main.cs"
    patch_once(main_file, "AppSettings.ReadOnly = !SavesData;",
               "AppSettings.ReadOnly = !SavesData;\n        if (Dev && StartAgentLab()) return;")
    for signature in ["public override void _Process(double delta)", "public override void _Input(InputEvent e)"]:
        anchor = signature + "\n    {"
        patch_once(main_file, anchor, anchor + "\n        if (AgentLabActive) return;")
    game_file = project / "src" / "Game" / "GameSession.cs"
    anchor = "public override void _Process(double delta)\n    {"
    patch_once(game_file, anchor, anchor + "\n        if (AgentControlled && !agentTicking) return;")
    anchor = "public override void _Input(InputEvent e)\n    {"
    patch_once(game_file, anchor, anchor + "\n        if (AgentControlled) return;")
    if args.source_only:
        print("Pinned upstream source and overlay prepared")
        return
    tool_dir = ROOT / ".tools" / ("godot-" + target_os)
    if not tool_dir.exists():
        unpack_zip(godot_zip, tool_dir)
    if target_os == "linux":
        godot = next(tool_dir.rglob("Godot_*linux.x86_64"))
        godot.chmod(godot.stat().st_mode | 0o111)
        dotnet_dir = ROOT / ".tools" / "dotnet-linux"
        dotnet = dotnet_dir / "dotnet"
        if not dotnet.exists():
            dotnet_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(sdk_tar) as source:
                if not hasattr(tarfile, "data_filter"):
                    raise RuntimeError("Upgrade Python to a version with safe tar extraction")
                source.extractall(dotnet_dir, filter="data")
        dotnet.chmod(dotnet.stat().st_mode | 0o111)
    else:
        godot = next(tool_dir.rglob("Godot_*_console.exe"))
        dotnet_command = shutil.which("dotnet")
        if not dotnet_command:
            raise RuntimeError("Install .NET 8 SDK for Windows, then rerun bootstrap")
        dotnet = Path(dotnet_command)
        dotnet_dir = dotnet.parent
    local = {"godot": str(godot.resolve()), "dotnet": str(dotnet.resolve()),
             "project": str(project.resolve()), "maps_dir": str(args.maps_dir.resolve()) if args.maps_dir else ""}
    if args.maps_dir and not (args.maps_dir / "maps").is_dir():
        raise ValueError("--maps-dir must contain a maps subdirectory")
    local_path = ROOT / "local.json"
    if local_path.exists() and not args.maps_dir:
        local["maps_dir"] = json.loads(local_path.read_text()).get("maps_dir", "")
    local_path.write_text(json.dumps(local, indent=2), encoding="utf-8")
    if args.skip_build:
        print("Source and tools prepared; build not run")
        return
    env = os.environ.copy()
    env.update(DOTNET_ROOT=str(dotnet_dir), DOTNET_CLI_TELEMETRY_OPTOUT="1",
               DOTNET_NOLOGO="1", PATH=str(dotnet_dir) + os.pathsep + env.get("PATH", ""))
    subprocess.run([str(dotnet), "build", str(project / "CaValAim.csproj"), "-c", "Debug", "--nologo"], env=env, check=True)
    # Import only: headless is appropriate here, never for rendered pixel observations.
    subprocess.run([str(godot), "--headless", "--editor", "--path", str(project), "--import", "--quit"], env=env, check=True)
    print("Bootstrap complete. Run: valorant-lab doctor")


if __name__ == "__main__":
    main()
