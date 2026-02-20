#!/usr/bin/env python3
"""Lightweight status server for runpod pod monitoring on port 8000."""

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _read_file_text(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return default


def _read_json(path: str) -> dict:
    text = _read_file_text(path, default="")
    if not text.strip():
        return {"status": "waiting", "message": "status file not available yet"}
    try:
        return json.loads(text)
    except Exception:
        return {"status": "invalid_status", "message": "status json parse error"}


def build_app(status_file: str, log_file: str, status_title: str, log_lines: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _set_headers(self, code=200, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/status"):
                status = _read_json(status_file)
                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(status).encode("utf-8"))
                return

            if self.path.startswith("/log"):
                text = _read_file_text(log_file)
                lines = text.splitlines()
                tail = "\n".join(lines[-log_lines:]) if lines else ""
                self._set_headers(200, "text/plain; charset=utf-8")
                self.wfile.write(tail.encode("utf-8"))
                return

            if self.path.startswith("/"):
                status = _read_json(status_file)
                status_html = html.escape(json.dumps(status, indent=2))
                html_body = f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta http-equiv=\"refresh\" content=\"5\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{html.escape(status_title)}</title>
    <style>
      body {{ font-family: ui-monospace, Menlo, Monaco, Consolas, monospace; padding: 16px; }}
      .box {{ border: 1px solid #ddd; padding: 12px; border-radius: 8px; max-width: 1000px; }}
      pre {{ overflow-x: auto; background: #111; color: #f8f8f8; padding: 12px; border-radius: 8px; }}
      a {{ display: inline-block; margin-right: 12px; }}
    </style>
  </head>
  <body>
    <h3>{html.escape(status_title)}</h3>
    <p>
      <a href=\"/status\">JSON Status</a>
      <a href=\"/log\">Tail Log</a>
    </p>
    <div class=\"box\">
      <pre id=\"status\">{status_html}</pre>
    </div>
  </body>
</html>
"""
                self._set_headers(200, "text/html; charset=utf-8")
                self.wfile.write(html_body.encode("utf-8"))
                return

            self._set_headers(404, "text/plain; charset=utf-8")
            self.wfile.write(b"not found")

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve shard status and log over HTTP")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--status-title", default="people-cleaner monitor")
    parser.add_argument("--log-lines", type=int, default=200)
    args = parser.parse_args()

    handler = build_app(
        status_file=args.status_file,
        log_file=args.log_file,
        status_title=args.status_title,
        log_lines=args.log_lines,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"monitor_server listening on 0.0.0.0:{args.port}")
    print(f"status_file={args.status_file}")
    print(f"log_file={args.log_file}")
    try:
        server.serve_forever()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
