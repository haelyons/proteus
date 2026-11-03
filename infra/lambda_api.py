"""Minimal Lambda Cloud + Jupyter contents client (stdlib only).

Auth to cloud.lambda.ai is injected by the environment's egress proxy, so the
bearer token here is a placeholder. The Jupyter token is a per-instance secret:
it is kept in memory only and never logged; URLs are logged with the query
string stripped.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://cloud.lambda.ai/api/v1"
UA = "esm-atlas/0.1"  # Cloudflare in front of the Lambda API rejects the default Python-urllib UA (error 1010)
_HDR = {"Authorization": "Bearer placeholder-injected-by-proxy", "Content-Type": "application/json", "User-Agent": UA}


def _req(method: str, url: str, body=None, headers=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw and r.headers.get_content_type() == "application/json" else raw)


def api(method: str, path: str, body=None, timeout=30):
    return _req(method, API + path, body, _HDR, timeout)[1]


def launch(**kw) -> str:
    return api("POST", "/instance-operations/launch", kw)["data"]["instance_ids"][0]


def get_instance(iid: str) -> dict:
    return api("GET", f"/instances/{iid}")["data"]


def list_instances() -> list[dict]:
    return api("GET", "/instances")["data"]


def terminate(ids: list[str]):
    return api("POST", "/instance-operations/terminate", {"instance_ids": ids})


def redact(url: str | None) -> str | None:
    if not url:
        return url
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))


class Jupyter:
    """Jupyter Server REST contents API over HTTPS."""

    def __init__(self, jupyter_url: str, token: str):
        p = urllib.parse.urlsplit(jupyter_url)
        self.base = f"{p.scheme}://{p.netloc}"
        self.host = p.netloc
        self._h = {"Authorization": f"token {token}", "Content-Type": "application/json", "User-Agent": UA}

    def get(self, path: str, content=True, timeout=30):
        q = "" if content else "?content=0"
        return _req("GET", f"{self.base}/api/contents/{urllib.parse.quote(path)}{q}", None, self._h, timeout)

    def read_text(self, path: str) -> str:
        _, d = self.get(path)
        if d.get("format") == "base64":
            import base64

            return base64.b64decode(d["content"]).decode(errors="replace")
        return d["content"]

    def put_text(self, path: str, text: str):
        body = {"type": "file", "format": "text", "content": text}
        return _req("PUT", f"{self.base}/api/contents/{urllib.parse.quote(path)}", body, self._h)

    def put_b64(self, path: str, data: bytes):
        import base64

        body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode()}
        return _req("PUT", f"{self.base}/api/contents/{urllib.parse.quote(path)}", body, self._h, timeout=120)


def put_file_chunked(jup: "Jupyter", path: str, local: str, chunk_bytes: int = 8 << 20, log=print):
    """Jupyter Server chunked upload (LargeFileManager): chunk numbers 1, 2, ..., last = -1."""
    import base64
    import os

    size = os.path.getsize(local)
    n = max(1, -(-size // chunk_bytes))
    t0 = now()
    if n == 1:  # chunk=-1 without a preceding chunk 1 is rejected (404): use a plain PUT
        with open(local, "rb") as f:
            jup.put_b64(path, f.read())
        _, meta = jup.get(path, content=False)
        assert int(meta.get("size", -1)) == size, (meta.get("size"), size)
        return size, now() - t0
    with open(local, "rb") as f:
        for i in range(1, n + 1):
            data = f.read(chunk_bytes)
            body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode(),
                    "chunk": -1 if i == n else i}
            _req("PUT", f"{jup.base}/api/contents/{urllib.parse.quote(path)}", body, jup._h, timeout=300)
            if i % 8 == 0 or i == n:
                log(f"upload {path}: {i}/{n} chunks, {i * chunk_bytes / max(1e-9, now() - t0) / 1e6:.1f} MB/s")
    _, meta = jup.get(path, content=False)
    assert int(meta.get("size", -1)) == size, (meta.get("size"), size)
    return size, now() - t0


def classify_error(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    s = str(getattr(e, "reason", e))
    if "Tunnel connection failed" in s:
        return f"proxy CONNECT refused ({s.split(':')[-1].strip()})"
    return type(e).__name__ + ": " + s[:120]


def now() -> float:
    return time.time()
