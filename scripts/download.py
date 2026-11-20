"""Verified, resumable HTTPS downloads. Standard library only."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path, expected: str, algorithm: str = "sha256") -> Path:
    if not url.startswith("https://"):
        raise ValueError("Downloads require HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if checksum(target, algorithm) != expected.lower():
            raise ValueError(f"Checksum mismatch: {target}; preserve and inspect this file")
        return target
    partial = target.with_name(target.name + ".part")
    # Respect explicitly configured proxies, with localhost always excluded in the RPC client.
    stalls = 0
    for attempt in range(120):
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={
            "User-Agent": "valorant-agent-lab/0.1", "Accept-Encoding": "identity",
            **({"Range": f"bytes={start}-"} if start else {})})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if start and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Unexpected download range")
                    mode = "ab"
                else:
                    mode = "wb"  # Server may not support Range (e.g. source archive).
                began = time.monotonic()
                with partial.open(mode) as output:
                    while True:
                        block = response.read1(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        # Short connections allow continuation through proxies with hard time limits.
                        if response.status == 206 and time.monotonic() - began > 45:
                            break
                total_header = response.headers.get("Content-Range", "").split("/")[-1]
                total = int(total_header) if total_header.isdigit() else None
                complete = total == partial.stat().st_size if total else not block
                if complete:
                    actual = checksum(partial, algorithm)
                    if actual != expected.lower():
                        raise ValueError(f"Checksum mismatch for {target.name}: {actual}")
                    os.replace(partial, target)
                    return target
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            print(f"Retry {target.name}: {type(error).__name__}", flush=True)
        now = partial.stat().st_size if partial.exists() else 0
        print(f"Downloaded {target.name}: {now / 1e6:.1f} MB", flush=True)
        stalls = stalls + 1 if now <= start else 0
        if stalls >= 6:
            raise RuntimeError(f"Download stalled: {target.name}; retry with a working network/proxy")
    raise RuntimeError(f"Download incomplete: {target}")
