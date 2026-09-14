"""HTTP server for the interactive viewer (stdlib only)."""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..config import CarConfig
from .session import ViewerSession, dispatch

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {"js": "text/javascript", "css": "text/css", "html": "text/html", "png": "image/png",
                 "svg": "image/svg+xml", "json": "application/json"}


class _Handler(BaseHTTPRequestHandler):
    session: ViewerSession = None  # type: ignore
    quiet = True

    def log_message(self, fmt, *args):  # pragma: no cover
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, ctype: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (STATIC / "index.html").read_bytes())
        elif path.startswith("/static/"):
            f = (STATIC / path[len("/static/"):]).resolve()
            if not str(f).startswith(str(STATIC.resolve())) or not f.is_file():
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, CONTENT_TYPES.get(f.suffix.lstrip("."), "application/octet-stream"), f.read_bytes())
        else:
            self._send(*dispatch(self.session, "GET", path))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except json.JSONDecodeError as e:
            self._send(400, "application/json", json.dumps({"error": f"bad json: {e}"}).encode())
            return
        self._send(*dispatch(self.session, "POST", urlparse(self.path).path, data))


def make_server(session: ViewerSession, host="127.0.0.1", port=8765, quiet=True) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"session": session, "quiet": quiet})
    return ThreadingHTTPServer((host, port), handler)


def serve(config_path=None, host="127.0.0.1", port=8765, open_browser=True, output_dir=None):
    cfg = CarConfig.load(config_path) if config_path else CarConfig.from_dict({"name": "untitled"})
    session = ViewerSession(cfg, output_dir=output_dir)
    httpd = make_server(session, host, port, quiet=False)
    url = f"http://{host}:{port}/"
    print(f"makecar viewer: {url}   (config: {cfg.path or 'unsaved'})   Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
