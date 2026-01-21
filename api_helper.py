import base64
from functools import wraps
from flask import request, jsonify
from settings import AUTH_USER, AUTH_PASS


def require_basic_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization")

        if not auth or not auth.startswith("Basic "):
            return jsonify({"error": "Unauthorized"}), 401

        try:
            encoded = auth.split(" ")[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return jsonify({"error": "Invalid auth header"}), 401

        if username != AUTH_USER or password != AUTH_PASS:
            return jsonify({"error": "Invalid credentials"}), 401

        return f(*args, **kwargs)

    return decorated
