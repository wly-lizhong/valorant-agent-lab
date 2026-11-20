from __future__ import annotations
import json
import socket


class RpcClient:
    """Local-only NDJSON RPC; bypasses HTTP proxy variables by using raw sockets."""

    def __init__(self, port: int, timeout: float = 60):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reader = self.socket.makefile("rb")
        self.sequence = 0

    def call(self, command: str, **payload) -> dict:
        self.sequence += 1
        request = {"id": self.sequence, "command": command, **payload}
        self.socket.sendall(json.dumps(request, allow_nan=False).encode() + b"\n")
        line = self.reader.readline(16 * 1024 * 1024 + 1)
        if not line or not line.endswith(b"\n") or len(line) > 16 * 1024 * 1024:
            raise RuntimeError("Engine response missing, incomplete or oversized")
        data = json.loads(line)
        if data.get("id") != self.sequence:
            raise RuntimeError("RPC response ID mismatch")
        if not data.get("ok"):
            raise RuntimeError("Engine: " + str(data.get("error", "unknown error")))
        return data

    def close(self):
        self.reader.close()
        self.socket.close()
