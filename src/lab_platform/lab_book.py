"""Serve the static Lab Book dashboard."""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


def default_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


class LabBookHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the repo root while making `/` open the Lab Book page."""

    def __init__(self, *args, directory: str | None = None, **kwargs) -> None:
        self.repo_root = Path(directory or default_root_dir()).resolve()
        super().__init__(*args, directory=str(self.repo_root), **kwargs)

    def translate_path(self, path: str) -> str:
        if path in ("/", "/index.html"):
            return str(self.repo_root / "docs" / "lab_book" / "index.html")
        return super().translate_path(path)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dir", default=str(default_root_dir()),
                        help="repository root to serve markdown/assets from")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    index = root / "docs" / "lab_book" / "index.html"
    if not index.exists():
        raise SystemExit(f"Lab Book index not found: {index}")

    handler = lambda *a, **kw: LabBookHandler(*a, directory=str(root), **kw)
    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        print(f"Lab Book: http://{args.host}:{args.port}/", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nLab Book stopped.", flush=True)


if __name__ == "__main__":
    main()
