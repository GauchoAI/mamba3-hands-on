"""Serve the static Lab Book dashboard."""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


def default_book_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "lab_book"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dir", default=str(default_book_dir()))
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not (root / "index.html").exists():
        raise SystemExit(f"Lab Book index not found: {root / 'index.html'}")

    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(root), **kw
    )
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Lab Book: http://{args.host}:{args.port}/", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nLab Book stopped.", flush=True)


if __name__ == "__main__":
    main()
