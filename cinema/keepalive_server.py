#!/usr/bin/env python3
"""
================================================================================
🛡️ KEEPALIVE & CLOUDFLARE QUICK TUNNEL DOWNLOAD SERVER 🛡️
File: bulk_generator/templates/keepalive_server.py
Purpose: Starts a local HTTP server, creates a zero-config Cloudflare tunnel,
         prints the public download link, and keeps the Colab/Kaggle session alive.
================================================================================
"""

import os
import re
import sys
import time
import threading
import subprocess
import http.server
import socketserver
from pathlib import Path

def start_download_and_keepalive(
    serve_directory: str = "/tmp/output",
    port: int = 8000,
    keepalive_hours: int = 4
):
    """
    Spins up an HTTP file server for `serve_directory`, exposes it via Cloudflare
    Quick Tunnel, and maintains an active heartbeat loop.
    """
    serve_path = Path(serve_directory)
    serve_path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(serve_path))

    # 1. Start HTTP file server in background thread
    class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass # Suppress noisy request logs

    httpd = socketserver.TCPServer(("", port), QuietHTTPHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    print(f"📦 Local HTTP Server serving '{serve_path}' on port {port}")

    # 2. Setup Cloudflare Quick Tunnel
    cf_bin = Path("/tmp/cloudflared")
    if not cf_bin.exists():
        print("⬇️ Fetching Cloudflare Quick Tunnel binary...")
        try:
            subprocess.run([
                "curl", "-fsSL",
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "-o", str(cf_bin)
            ], check=True)
            cf_bin.chmod(0o755)
        except Exception as e:
            print(f"⚠️ Could not download cloudflared: {e}")

    tunnel_url = None
    cf_proc = None

    if cf_bin.exists():
        print("🌐 Establishing Cloudflare Quick Tunnel...")
        cf_proc = subprocess.Popen(
            [str(cf_bin), "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        start_time = time.time()
        while time.time() - start_time < 35:
            line = cf_proc.stdout.readline()
            if not line:
                break
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                tunnel_url = match.group(0)
                break

    print("\n" + "=" * 70)
    if tunnel_url:
        print(f"🎉 DIRECT DOWNLOAD LINK: {tunnel_url}")
        print("   (Click the link to browse and download your completed files!)")
    else:
        print(f"⚠️ Direct tunnel unavailable. Files available locally on port {port}")
    print("=" * 70 + "\n")

    # 3. Keepalive loop
    total_seconds = keepalive_hours * 3600
    interval = 60
    elapsed = 0

    print(f"🛡️ Keepalive daemon active for up to {keepalive_hours} hours.")
    print("   Press Ctrl+C in your notebook/terminal to terminate when done.\n")

    try:
        while elapsed < total_seconds:
            time.sleep(interval)
            elapsed += interval
            remaining_min = (total_seconds - elapsed) // 60
            if elapsed % 300 == 0: # Every 5 minutes
                print(f"💓 [Keepalive Heartbeat] Session active ({remaining_min}m remaining). Download: {tunnel_url}")
    except KeyboardInterrupt:
        print("\n🛑 Keepalive loop stopped by user.")
    finally:
        if cf_proc:
            cf_proc.terminate()
        httpd.shutdown()
        print("✅ Server cleanly shut down.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    start_download_and_keepalive(serve_directory=target)
