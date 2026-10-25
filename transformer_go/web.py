"""Local browser interface for human-vs-engine Go games."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from .play import PlayService, discover_weights


STATIC = Path(__file__).with_name("static")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def make_handler(service, host, port):
    class Handler(BaseHTTPRequestHandler):
        server_version = "transformerGo/0.1"

        def _headers(self, status, content_type, length, extra=()):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:")
            for name, value in extra:
                self.send_header(name, value)
            self.end_headers()

        def _json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _local_request(self):
            host_header = self.headers.get("Host", "")
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
            if host_header not in allowed_hosts:
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or parsed.netloc not in allowed_hosts:
                    return False
            return True

        def do_GET(self):
            if not self._local_request():
                self._json(403, {"error": "local requests only"})
                return
            path = urlsplit(self.path).path
            if path == "/api/weights":
                self._json(200, {"weights": service.weights})
                return
            if path == "/api/state":
                with service.lock:
                    self._json(200, service.state())
                return
            if path == "/api/sgf":
                with service.lock:
                    body = service.sgf().encode("utf-8")
                self._headers(200, "application/x-go-sgf; charset=utf-8", len(body),
                              (("Content-Disposition", 'attachment; filename="transformer-go-game.sgf"'),))
                self.wfile.write(body)
                return
            asset = ASSETS.get(path)
            if asset is None:
                self._json(404, {"error": "not found"})
                return
            filename, content_type = asset
            body = (STATIC / filename).read_bytes()
            self._headers(200, content_type, len(body))
            self.wfile.write(body)

        def do_POST(self):
            if not self._local_request():
                self._json(403, {"error": "local requests only"})
                return
            length = self.headers.get("Content-Length", "")
            try:
                size = int(length)
            except ValueError:
                self._json(400, {"error": "invalid content length"})
                return
            if size < 0 or size > 65536:
                self._json(413, {"error": "request body too large"})
                return
            try:
                payload = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                path = urlsplit(self.path).path
                with service.lock:
                    result = service.request(path, payload)
                self._json(200, result)
            except (json.JSONDecodeError, ValueError, TypeError, KeyError, RuntimeError) as error:
                self._json(400, {"error": str(error)})
            except Exception as error:
                self._json(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, fmt, *args):
            print(f"[play] {self.address_string()} {fmt % args}")

    return Handler


def serve(config, device, root, *, checkpoint=None, host="127.0.0.1", port=8765):
    try:
        is_loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("the play server only supports a loopback host")
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1,65535]")
    weights, default_id = discover_weights(root, checkpoint)
    service = PlayService(config, device, root, weights, default_id)
    server = ThreadingHTTPServer((host, port), make_handler(service, host, port))
    print(f"Human vs engine: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping human-vs-engine server", flush=True)
    finally:
        server.server_close()
