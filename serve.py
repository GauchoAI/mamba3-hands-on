"""No-cache HTTP server. Serves files with cache-busting headers."""
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 16006))  # Replace TensorBoard (Caddy proxies 6006→16006)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("0.0.0.0", port), NoCacheHandler)
    print(f"Serving on port {port} (no-cache)", flush=True)
    server.serve_forever()
