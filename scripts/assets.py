"""Pack a local map or fetch a user-hosted bundle; never publish/upload assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
from bootstrap import ROOT, unpack_zip
from download import checksum, fetch


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--root", type=Path, required=True)
    pack.add_argument("--map", default="lotus")
    pack.add_argument("--output", type=Path, default=ROOT / "assets/lotus-assets.zip")
    get = sub.add_parser("fetch")
    get.add_argument("--url", required=True)
    get.add_argument("--sha256", required=True)
    get.add_argument("--destination", type=Path, default=ROOT / "assets")
    install = sub.add_parser("install", help="Verify and unpack a local bundle")
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--sha256", required=True)
    install.add_argument("--destination", type=Path, default=ROOT / "assets/maps-data")
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    if args.command == "pack":
        if Path(args.map).name != args.map or "/" in args.map or "\\" in args.map:
            parser.error("Invalid map name")
        folder = args.root / "maps" / args.map
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file() or not (folder / "collision.json").is_file():
            parser.error("Map manifest/collision missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        names = ["manifest.json", "collision.json"]
        if (folder / "spawn_exclude.json").exists():
            names.append("spawn_exclude.json")
        for level in manifest["levels"]:
            if Path(level).name != level or "/" in level or "\\" in level:
                parser.error("Invalid level in map manifest")
            names.append(level + ".glb")
        inventory = {f"maps/{args.map}/{name}": checksum(folder / name) for name in sorted(names)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for relative in inventory:
                archive.write(args.root / relative, relative)
            archive.writestr("asset-inventory.json", json.dumps({"map": args.map, "files": inventory}, indent=2))
        digest = checksum(args.output)
        args.output.with_suffix(".sha256").write_text(digest + "  " + args.output.name + "\n")
        print(json.dumps({"bundle": str(args.output), "sha256": digest, "bytes": args.output.stat().st_size}, indent=2))
    elif args.command in ("fetch", "install"):
        if len(args.sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.sha256):
            parser.error("Expected a SHA-256 hex digest")
        if args.command == "fetch":
            archive = fetch(args.url, ROOT / ".cache" / (args.sha256 + ".zip"), args.sha256)
        else:
            archive = args.archive
            if checksum(archive) != args.sha256.lower():
                parser.error("Local asset bundle checksum mismatch")
        if args.destination.exists() and any(args.destination.iterdir()):
            parser.error("Asset destination is not empty; preserve it and choose another path")
        unpack_zip(archive, args.destination)
        verify_inventory(args.destination)
        print("Asset bundle verified. Pass this directory as --maps-dir to bootstrap.")
    else:
        verify_inventory(args.root)


def verify_inventory(root: Path):
    manifest = json.loads((root / "asset-inventory.json").read_text())
    for relative, expected in manifest["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or checksum(target) != expected:
            raise ValueError("Asset mismatch: " + relative)
    print(f"Verified {len(manifest['files'])} map files")


if __name__ == "__main__":
    main()
