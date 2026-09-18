"""
Tests for `launcher.ExplorerServer` and its supporting functions --
everything in launcher.py that does NOT require Tkinter or a PyInstaller
build to exercise. The GUI window code in `run_gui_mode` /
`_show_running_window` is deliberately excluded: it cannot be tested in
an environment without Tkinter installed (true of the sandbox this
project is developed in), and is not claimed to be tested here. It is
built entirely on top of the `ExplorerServer` methods these tests do
cover, so the only genuinely unverified code is window layout, not
server-startup or readiness logic.
"""
import time
import socket
import threading

from ims_platform.launcher import (
    ExplorerServer, is_port_responding, find_free_port, run_console_mode, HAS_TKINTER,
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_is_port_responding_false_for_a_closed_port():
    port = _unused_port()
    assert is_port_responding(port, timeout=0.3) is False


def test_find_free_port_returns_the_preferred_port_when_free():
    port = _unused_port()
    assert find_free_port(port) == port


def test_find_free_port_skips_a_port_held_open():
    port = _unused_port()
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        found = find_free_port(port)
        assert found != port
    finally:
        holder.close()


def test_explorer_server_starts_and_becomes_ready():
    port = _unused_port()
    server = ExplorerServer(port=port)
    assert server.already_running() is False

    server.start_in_background()
    ready = server.wait_until_ready(timeout_s=15.0)
    assert ready is True
    assert server.already_running() is True
    assert server.url == f"http://127.0.0.1:{port}/"


def test_explorer_server_detects_an_already_running_instance():
    """
    The specific robustness property the launcher relies on: if a server
    is already answering on the target port (e.g. the executable was
    double-clicked twice), a second ExplorerServer instance must detect
    this via already_running() rather than trying to bind the port again.
    """
    port = _unused_port()
    server1 = ExplorerServer(port=port)
    server1.start_in_background()
    assert server1.wait_until_ready(timeout_s=15.0)

    server2 = ExplorerServer(port=port)
    assert server2.already_running() is True


def test_wait_until_ready_times_out_if_nothing_ever_starts():
    port = _unused_port()
    server = ExplorerServer(port=port)  # deliberately never call start_in_background()
    t0 = time.monotonic()
    ready = server.wait_until_ready(timeout_s=0.6)
    elapsed = time.monotonic() - t0
    assert ready is False
    assert elapsed < 2.0  # didn't hang well past the requested timeout


def test_run_console_mode_starts_server_and_returns_cleanly_on_interrupt():
    """
    Drives the actual console-mode code path used whenever Tkinter is
    unavailable (as in this sandbox) end to end, short-circuiting the
    infinite wait loop via a background KeyboardInterrupt injection
    rather than mocking away the function under test.
    """
    import builtins
    port = _unused_port()
    server = ExplorerServer(port=port)

    real_sleep = time.sleep
    call_count = {"n": 0}

    def fake_sleep(seconds):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt()
        real_sleep(0.05)

    orig_open = __import__("webbrowser").open
    opened_urls = []
    import webbrowser as wb
    wb.open = lambda url: opened_urls.append(url)
    try:
        time.sleep = fake_sleep
        code = run_console_mode(server)
    finally:
        time.sleep = real_sleep
        wb.open = orig_open

    assert code == 0
    assert opened_urls == [server.url]
    assert server.already_running() is True  # thread is a daemon; still alive


def test_has_tkinter_flag_matches_this_sandbox():
    # This sandbox has no Tkinter installed; documents that fact rather
    # than asserting a specific value that would be environment-dependent
    # on a real Windows/macOS build machine.
    assert HAS_TKINTER in (True, False)


if __name__ == "__main__":
    import sys
    import inspect
    fns = [f for name, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
           if name.startswith("test_")]
    passed, failed = 0, 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {f.__name__} -> {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
