"""Immutable, hash-checked GitHub Release bundles. Never merge recordings with Git."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from .env import ROOT
from .dataset import sha256, write_json, DemonstrationDataset
from .corrections import load_case
from .scenario import validate_scenario, digest, contract

SCHEMA = "valorant-exchange-v1"
DEFAULT_REPO = "wly-lizhong/valorant-agent-lab"
MAX_BUNDLE = 1800 * 1024**2
MAX_EXPANDED = 4 * 1024**3


def upstream():
    return json.loads((ROOT / "dependencies.lock.json").read_text())["upstream"]["sha256"]


def validate_item(directory, kind):
    if kind == "tasks":
        if not re.fullmatch(r"[0-9a-f]{16}", directory.name):
            raise ValueError("Invalid correction ID")
        m = load_case(directory / "case.json")
        validate_scenario(m["scenario"])
        if (m["id"] != directory.name or m.get("replay") != "restart_from_seed"
                or type(m.get("engine_seed")) is not int or not 0 <= m["engine_seed"] < 2**31):
            raise ValueError("Invalid correction seed/replay")
        return m, ["case.json"] + [n for n in ("failure-context.json", "failure.png") if (directory / n).is_file()]
    if not re.fullmatch(r"episode-[0-9a-f]{16}", directory.name):
        raise ValueError("Invalid episode ID")
    m = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    validate_scenario(m["scenario"])
    if (m.get("episode_id") != directory.name or m.get("accepted") is not True or m.get("source") != "human"
            or m.get("schema") != "valorant-demo-v1" or m.get("complete") is not True or m.get("quality_issues")
            or m.get("final", {}).get("is_success") is not True
            or m.get("split") not in ("train", "validation") or m.get("contract") != contract(m["scenario"])
            or type(m.get("engine_seed")) is not int or not 0 <= m["engine_seed"] < 2**31
            or m.get("scenario_hash") != digest(m["scenario"])
            or m.get("case_key") != digest([m["scenario"], m["engine_seed"]])):
        raise ValueError("Only complete accepted human demonstrations may be exchanged")
    files = ["episode.json", "diagnostics.jsonl"]
    count = 0
    for index, chunk in enumerate(m["chunks"]):
        name = f"chunk-{index:05d}.npz"
        if chunk["file"] != name or not 0 < chunk["steps"] <= 256 or sha256(directory / name) != chunk["sha256"]:
            raise ValueError("Invalid demonstration chunk")
        # Bound expanded NPZ size before numpy reads it; object arrays are never allowed.
        with zipfile.ZipFile(directory / name) as z:
            expected = chunk["steps"] * (math.prod(m['contract']['shape']) + 8 * 8) + 8192
            if sum(x.file_size for x in z.infolist()) > expected:
                raise ValueError("Oversized demonstration tensor archive")
        files.append(name); count += chunk["steps"]
    if count != m["steps"] or count < 1 or count > m["scenario"]["max_steps"]:
        raise ValueError("Invalid demonstration length")
    dataset = DemonstrationDataset([(directory, m)])
    offset = 0
    for c in m["chunks"]:
        dataset[offset]  # validates every complete chunk without retaining the whole dataset
        offset += c["steps"]
    return m, files


def pack(directory, kind, output):
    directory, output = Path(directory), Path(output)
    if kind not in ("tasks", "demos"):
        raise ValueError("Unknown exchange kind")
    pattern = "*/case.json" if kind == "tasks" else "episode-*/episode.json"
    entries = []
    for path in sorted(directory.glob(pattern)):
        if kind == "demos" and not json.loads(path.read_text(encoding="utf-8")).get("accepted"):
            continue
        _, names = validate_item(path.parent, kind)
        for name in names:
            source = path.parent / name
            if source.is_symlink() or not source.is_file():
                raise ValueError("Bundle files must be ordinary files")
            entries.append((source, f"items/{path.parent.name}/{name}"))
    if not entries:
        raise ValueError("没有可交换的数据；示范必须成功且经人工确认")
    manifest = {"schema": SCHEMA, "kind": kind, "upstream_sha256": upstream(),
                "files": [{"path": name, "size": p.stat().st_size, "sha256": sha256(p)} for p, name in entries]}
    if sum(x["size"] for x in manifest["files"]) > MAX_EXPANDED:
        raise ValueError("数据过多，请按批次打包")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr("bundle.json", json.dumps(manifest, ensure_ascii=False))
        for (p, name), meta in zip(entries, manifest["files"]):
            z.write(p, name)
            if sha256(p) != meta["sha256"]:
                raise ValueError("Source changed while packing; retry after recording finishes")
    if output.stat().st_size > MAX_BUNDLE:
        raise ValueError("附件超过 1800 MiB，请拆成多批")
    output.with_suffix(output.suffix + ".sha256").write_text(sha256(output) + "  " + output.name + "\n", encoding="ascii")
    return output


def unpack(archive, destination, expected_kind=None):
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size > MAX_BUNDLE:
        raise ValueError("Bundle too large")
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    if not checksum_path.is_file() or checksum_path.read_text().split()[0] != sha256(archive):
        raise ValueError("Missing/mismatched bundle SHA-256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".valab-import-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            if (len(infos) > 100000 or sum(x.file_size for x in infos) > MAX_EXPANDED
                    or len({x.filename for x in infos}) != len(infos)):
                raise ValueError("Invalid archive size or duplicate names")
            if z.getinfo("bundle.json").file_size > 8 * 1024**2:
                raise ValueError("Oversized bundle manifest")
            m = json.loads(z.read("bundle.json"))
            kind = m.get("kind")
            if (m.get("schema") != SCHEMA or kind not in ("tasks", "demos")
                    or expected_kind and kind != expected_kind or m.get("upstream_sha256") != upstream()):
                raise ValueError("Bundle format/kind/upstream version mismatch")
            names = [f["path"] for f in m["files"]]
            if len(set(names)) != len(names) or set(names) | {"bundle.json"} != {i.filename for i in infos}:
                raise ValueError("Archive inventory mismatch")
            for meta in m["files"]:
                name = meta["path"]
                if not re.fullmatch(r"items/(?:episode-)?[0-9a-f]{16}/(?:case\.json|failure-context\.json|failure\.png|episode\.json|diagnostics\.jsonl|chunk-[0-9]{5}\.npz)", name):
                    raise ValueError("Unsafe archive path")
                info = z.getinfo(name)
                if info.file_size != meta["size"] or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Invalid archive member")
                target = stage / name; target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(target) != meta["sha256"]:
                    raise ValueError("Corrupt archive member")
        items = sorted((stage / "items").iterdir())
        checked = [(item, *validate_item(item, kind)) for item in items]
        for item, _, allowed in checked:
            if set(allowed) != {p.name for p in item.iterdir()}:
                raise ValueError("Unexpected bundle contents")
        # Validate the entire batch, existing IDs and split separation before publishing any directory.
        groups = {}
        if kind == "demos":
            for p in destination.glob("episode-*/episode.json"):
                old = json.loads(p.read_text(encoding="utf-8"))
                if old.get("accepted"):
                    prior = groups.setdefault(old["case_key"], old["split"])
                    if prior != old["split"]: raise ValueError("Existing train/validation leakage")
        new_items, skipped = [], 0
        for item, meta, files in checked:
            target = destination / item.name
            if kind == "demos":
                excluded_root = destination / ".excluded"
                excluded = excluded_root / item.name
                if excluded_root.is_symlink() or excluded.is_symlink():
                    raise ValueError("Excluded demonstrations may not be symlinks")
                if excluded.exists():
                    if target.exists() or target.is_symlink():
                        raise ValueError("Episode exists in both active and excluded data")
                    _, excluded_files = validate_item(excluded, kind)
                    if excluded_files != files or any(sha256(excluded / f) != sha256(item / f) for f in files):
                        raise ValueError("Conflicting excluded ID; preserving excluded data")
                    # Keep explicit exclusions across cumulative/repeated uploads. The
                    # raw files remain unchanged, outside the loader/exporter's glob.
                    skipped += 1
                    continue
                prior = groups.setdefault(meta["case_key"], meta["split"])
                if prior != meta["split"]: raise ValueError("Train/validation leakage")
            if target.is_symlink(): raise ValueError("Destination may not be a symlink")
            if target.exists():
                old, old_files = validate_item(target, kind)
                same = (old["scenario"] == meta["scenario"] and old["engine_seed"] == meta["engine_seed"]) if kind == "tasks" else (old_files == files and all(sha256(target / f) == sha256(item / f) for f in files))
                if not same: raise ValueError("Conflicting existing ID; preserving existing data")
                skipped += 1
            else: new_items.append((item, target))
        destination.mkdir(parents=True, exist_ok=True)
        for source, target in new_items:
            if target.exists(): raise FileExistsError(target)
            source.rename(target)
        return {"kind": kind, "imported": len(new_items), "skipped": skipped}


def gh(*args):
    if shutil.which("gh") is None:
        raise RuntimeError("请先安装 GitHub CLI（https://cli.github.com），再运行 gh auth login 登录。")
    try:
        return subprocess.check_output(["gh", *map(str, args)], text=True, encoding="utf-8",
                                       stderr=subprocess.PIPE, timeout=1800)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("GitHub 操作未完成；请检查登录、仓库权限和网络。\n" + error.stderr[-1500:]) from error


def push(directory, kind, repo=DEFAULT_REPO):
    tag = f"{kind}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    work = ROOT / "runs/exchange" / tag; work.mkdir(parents=True)
    archive = pack(directory, kind, work / (tag + ".valab.zip"))
    # Explicit SHA: never let a release tag drift to an unrelated default-branch commit.
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    notes = work / "release-notes.txt"
    notes.write_text("跨平台训练数据交换。导入前验证 SHA-256、上游版本和训练/验证分组。\n", encoding="utf-8")
    gh("release", "create", tag, archive, str(archive) + ".sha256", "--repo", repo,
       "--target", sha, "--prerelease", "--latest=false", "--title", f"{kind} 数据交换 {tag}", "--notes-file", notes)
    return tag


def pull(destination, kind, repo=DEFAULT_REPO, tag=None):
    if tag is None:
        releases = json.loads(gh("release", "list", "--repo", repo, "--limit", "1000", "--json", "tagName,isDraft"))
        tags = [r["tagName"] for r in reversed(releases) if not r["isDraft"] and r["tagName"].startswith(kind + "-")]
        if not tags: raise ValueError("GitHub 暂无对应的数据包")
        total = {"kind": kind, "imported": 0, "skipped": 0, "batches": 0}
        receipts = Path(destination) / ".exchange-receipts"
        for batch in tags:
            receipt = receipts / (digest([repo, batch]) + ".json")
            if receipt.is_file(): continue
            result = pull(destination, kind, repo, batch)
            receipts.mkdir(parents=True, exist_ok=True)
            write_json(receipt, {"repo": repo, "tag": batch, **result})
            for key in ("imported", "skipped"): total[key] += result[key]
            total["batches"] += 1
        return total
    if not tag or not re.fullmatch(r"(?:tasks|demos)-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", tag) or not tag.startswith(kind + "-"):
        raise ValueError("未找到对应的数据 Release；也可用 --tag 指定批次")
    work = ROOT / "runs/exchange/downloads" / (tag + "-" + uuid.uuid4().hex[:6]); work.mkdir(parents=True)
    gh("release", "download", tag, "--repo", repo, "--dir", work,
       "--pattern", "*.valab.zip", "--pattern", "*.valab.zip.sha256")
    archives = list(work.glob("*.valab.zip"))
    if len(archives) != 1: raise ValueError("Expected exactly one bundle")
    return unpack(archives[0], destination, kind)


def main():
    p = argparse.ArgumentParser(description="纠错题 / 示范的跨平台交换；公开仓库的附件同样公开")
    p.add_argument("action", choices=["pack", "unpack", "push", "pull"])
    p.add_argument("--kind", choices=["tasks", "demos"], required=True)
    p.add_argument("--directory", required=True)
    p.add_argument("--archive")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--tag")
    a = p.parse_args()
    if a.action in ("pack", "unpack") and not a.archive: p.error("--archive required")
    if a.action == "pack": result = pack(a.directory, a.kind, a.archive)
    elif a.action == "unpack": result = unpack(a.archive, a.directory, a.kind)
    elif a.action == "push": result = push(a.directory, a.kind, a.repo)
    else: result = pull(a.directory, a.kind, a.repo, a.tag)
    print(result)


if __name__ == "__main__": main()
