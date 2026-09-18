"""
Minimal local authentication for the IMS Platform Explorer.

Not a production-grade auth system: no email verification, no password
reset, no rate limiting, no HTTPS-only cookie enforcement (the app is
served over plain HTTP in local/dev use, e.g. GitHub Codespaces). It
exists to make the "Project Manager" header control genuinely
functional (sign up, sign in, sign out) rather than decorative, per an
explicit user request -- not to be a secure multi-tenant identity
system. Passwords are stored salted and hashed (PBKDF2-HMAC-SHA256),
never in plaintext, and session tokens are opaque random values kept
server-side, not JWTs with embedded claims -- both real security
practices even though the overall system is intentionally minimal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

_USERS_FILE = Path(os.environ.get("IMS_USERS_FILE", Path(__file__).resolve().parent.parent.parent / "data" / "users.json"))
_SESSIONS: dict[str, dict] = {}  # token -> {"username": ..., "created_at": ...}
_SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")


def _load_users() -> dict:
    if not _USERS_FILE.exists():
        return {}
    try:
        return json.loads(_USERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USERS_FILE.write_text(json.dumps(users, indent=2))


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex() + ":" + digest.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split(":", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = _hash_password(password, salt)
    return hmac.compare_digest(expected, stored)


def _prune_expired_sessions() -> None:
    now = time.time()
    expired = [t for t, s in _SESSIONS.items() if now - s["created_at"] > _SESSION_TTL_SECONDS]
    for t in expired:
        del _SESSIONS[t]


def _current_username() -> Optional[str]:
    _prune_expired_sessions()
    token = request.headers.get("X-Session-Token") or (request.get_json(silent=True) or {}).get("session_token")
    if not token or token not in _SESSIONS:
        return None
    return _SESSIONS[token]["username"]


def register_auth_routes(app: Flask) -> None:
    @app.route("/api/auth/signup", methods=["POST"])
    def auth_signup():
        payload = request.get_json(force=True, silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if not _USERNAME_RE.match(username):
            return jsonify({"error": "Username must be 3-32 characters: letters, digits, underscore, dot, or hyphen only."}), 400
        if len(password) < 8:
            return jsonify({"error": "Password must be at least 8 characters."}), 400
        users = _load_users()
        key = username.lower()
        if key in users:
            return jsonify({"error": "That username is already taken."}), 409
        users[key] = {
            "username": username, "password_hash": _hash_password(password),
            "created_at": time.time(),
        }
        _save_users(users)
        token = secrets.token_urlsafe(32)
        _SESSIONS[token] = {"username": username, "created_at": time.time()}
        return jsonify({"session_token": token, "username": username})

    @app.route("/api/auth/login", methods=["POST"])
    def auth_login():
        payload = request.get_json(force=True, silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        users = _load_users()
        rec = users.get(username.lower())
        if not rec or not _verify_password(password, rec["password_hash"]):
            return jsonify({"error": "Incorrect username or password."}), 401
        token = secrets.token_urlsafe(32)
        _SESSIONS[token] = {"username": rec["username"], "created_at": time.time()}
        return jsonify({"session_token": token, "username": rec["username"]})

    @app.route("/api/auth/logout", methods=["POST"])
    def auth_logout():
        payload = request.get_json(force=True, silent=True) or {}
        token = payload.get("session_token") or request.headers.get("X-Session-Token")
        if token and token in _SESSIONS:
            del _SESSIONS[token]
        return jsonify({"ok": True})

    @app.route("/api/auth/me", methods=["GET", "POST"])
    def auth_me():
        username = _current_username()
        if username is None:
            return jsonify({"authenticated": False}), 200
        return jsonify({"authenticated": True, "username": username})
