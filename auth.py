"""Supabase access-token verification for the Flask back end.

The project signs tokens with an asymmetric key (ES256), so they are verified
locally against the project's public JWKS — no secret key, and no network call
per request once the keys are cached. Lives outside src/ because it imports Flask.
"""
import os
from functools import wraps

import jwt
from flask import g, jsonify, request

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or ""

AUDIENCE = "authenticated"
# Pinned so a token can't pick its own algorithm (e.g. HS256 keyed with the public key).
ALGORITHMS = ["ES256"]
JWKS_CACHE_S = 600

_jwks = None
if SUPABASE_URL:
    # Caches the key set; an unknown `kid` (after a key rotation) triggers a refetch.
    _jwks = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
                            cache_jwk_set=True, lifespan=JWKS_CACHE_S)


def auth_configured():
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)


def verify_token(token):
    """Decoded claims for a valid token; raises jwt.PyJWTError otherwise."""
    key = _jwks.get_signing_key_from_jwt(token).key
    return jwt.decode(
        token, key,
        algorithms=ALGORITHMS,
        audience=AUDIENCE,
        issuer=f"{SUPABASE_URL}/auth/v1",
        options={"require": ["exp", "sub", "aud", "iss"]},
    )


def require_user(view):
    """Reject the request unless it carries a valid Supabase access token.

    The verified claims are available to the view as `g.user`.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        if _jwks is None:
            return jsonify({"error": "Login is not configured on this server."}), 503

        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return jsonify({"error": "Missing bearer token."}), 401

        try:
            g.user = verify_token(token.strip())
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired. Please log in again."}), 401
        except jwt.PyJWKClientConnectionError:
            return jsonify({"error": "Could not reach the auth server "
                                     "to verify the session."}), 503
        except jwt.PyJWTError:
            return jsonify({"error": "Invalid access token."}), 401
        return view(*args, **kwargs)
    return wrapper
