"""
launcher
--------

The entry point for the standalone "IMS Platform Explorer" executable
(built via PyInstaller, see build_scripts/). Responsible for exactly the
user experience requested:

    Double-click executable
        -> splash screen ("Starting IMS Platform Explorer...")
        -> browser opens automatically once the backend is ready
        -> Project Manager appears

Design notes
------------
- Runs in one of two modes, chosen automatically at import time:
  a Tkinter GUI splash/status window if Tkinter is importable on the
  target machine (true for the standard Python installers PyInstaller
  bundles on Windows and macOS), or a console-only banner + blocking
  loop otherwise (true of this development sandbox, and of some minimal
  Linux Python installations). The two modes share the same underlying
  server-startup and readiness-polling logic (`ExplorerServer` below),
  which is what this repository's test suite actually exercises --
  the Tkinter-specific window code cannot be exercised in an
  environment without Tkinter installed, and is not claimed to be.
- The Flask development server (see server.app) runs in a background
  daemon thread; readiness is determined by polling `/api/health` with
  a real HTTP request via `urllib`, not by a fixed sleep.
- If a server is already running on the target port (e.g. the user
  double-clicked the executable twice), the launcher detects this via
  the same health check and simply opens a new browser tab against the
  existing instance rather than starting a second server.
"""

from __future__ import annotations

import os
import sys
import time
import socket
import threading
import urllib.request
import urllib.error
import webbrowser

try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except Exception:
    HAS_TKINTER = False

DEFAULT_PORT = 8765
HEALTH_TIMEOUT_S = 20.0
POLL_INTERVAL_S = 0.25


def resource_base_dir() -> str:
    """
    Directory to resolve bundled resources (notably server/static/*.html)
    relative to. When frozen by PyInstaller, resources are unpacked to
    `sys._MEIPASS`; in a normal source checkout, it's this file's own
    directory (the ims_platform package root).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


def is_port_responding(port: int, path: str = "/api/health", timeout: float = 0.5) -> bool:
    """True if something is already answering HTTP on this port with a 2xx/3xx response."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def find_free_port(preferred: int, max_tries: int = 20) -> int:
    """
    Return `preferred` if free, else the next free port. Does not itself
    detect "an Explorer instance is already running here" -- that is
    `is_port_responding`'s job, checked separately and first, so the
    launcher can *reuse* an existing instance rather than always hunting
    for a new port.
    """
    port = preferred
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"could not find a free port after {max_tries} tries starting at {preferred}")


class ExplorerServer:
    """
    Owns starting the Flask app in a background thread and polling for
    readiness. Contains no GUI code, so it is fully unit-testable
    without Tkinter (see tests/test_launcher.py).
    """

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self._thread: threading.Thread | None = None

    def already_running(self) -> bool:
        return is_port_responding(self.port)

    def start_in_background(self) -> None:
        from .server.app import create_app

        app = create_app()

        def _run():
            # use_reloader=False is required: the reloader forks a second
            # process, which would break the "one background thread" model
            # this launcher relies on, and is meaningless in a frozen exe anyway.
            app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if is_port_responding(self.port):
                return True
            time.sleep(POLL_INTERVAL_S)
        return False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


def run_console_mode(server: ExplorerServer) -> int:
    """Console-only launch path (no Tkinter available). Returns a process exit code."""
    print("=" * 60)
    print("  IMS Platform Explorer")
    print("=" * 60)

    if server.already_running():
        print(f"  An instance is already running at {server.url}")
        print("  Opening it in your browser...")
        webbrowser.open(server.url)
        return 0

    print("  Starting IMS Platform Explorer...")
    server.start_in_background()

    if not server.wait_until_ready():
        print(f"  ERROR: backend did not become ready within {HEALTH_TIMEOUT_S:.0f}s.")
        print("  Check for a port conflict or consult the log output above.")
        return 1

    print(f"  Ready. Opening {server.url} in your browser...")
    webbrowser.open(server.url)
    print()
    print("  IMS Platform Explorer is running.")
    print("  Press Ctrl+C in this window to stop the server.")
    print("=" * 60)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  Stopping IMS Platform Explorer.")
        return 0


def run_gui_mode(server: ExplorerServer) -> int:
    """
    Tkinter GUI launch path: a splash window while the backend starts,
    replaced by a small persistent status window once it is ready.
    Not exercised by the automated test suite (no Tkinter in the
    development/build sandbox) -- kept deliberately simple and built
    entirely on the same ExplorerServer methods run_console_mode uses,
    so the only genuinely untested code here is window layout, not
    server logic.
    """
    root = tk.Tk()
    root.title("IMS Platform Explorer")
    root.geometry("380x160")
    root.resizable(False, False)
    root.eval("tk::PlaceWindow . center")

    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    title = ttk.Label(frame, text="IMS Platform Explorer", font=("", 13, "bold"))
    title.pack(pady=(0, 8))
    status = ttk.Label(frame, text="Starting IMS Platform Explorer...")
    status.pack(pady=(0, 12))
    progress = ttk.Progressbar(frame, mode="indeterminate", length=280)
    progress.pack()
    progress.start(12)

    already = server.already_running()
    if not already:
        server.start_in_background()

    result = {"ready": False}

    def poll():
        if is_port_responding(server.port):
            result["ready"] = True
            progress.stop()
            _show_running_window(root, frame, title, status, progress, server)
        else:
            root.after(int(POLL_INTERVAL_S * 1000), poll)

    root.after(50, poll)
    root.mainloop()
    return 0 if result["ready"] else 1


def _show_running_window(root, frame, title, status, progress, server: ExplorerServer):
    progress.destroy()
    status.config(text=f"Running at {server.url}")
    webbrowser.open(server.url)

    hint = ttk.Label(frame, text="Close this window to stop the server.", foreground="#666666")
    hint.pack(pady=(4, 12))
    quit_btn = ttk.Button(frame, text="Quit", command=root.destroy)
    quit_btn.pack()


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    port = DEFAULT_PORT
    if argv and argv[0].isdigit():
        port = int(argv[0])

    server = ExplorerServer(port=port)
    if HAS_TKINTER:
        return run_gui_mode(server)
    return run_console_mode(server)


if __name__ == "__main__":
    sys.exit(main())
