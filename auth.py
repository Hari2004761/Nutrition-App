"""Supabase login for the Flask back end, with tokens kept in HttpOnly cookies.

The browser never sees a token: Flask calls Supabase Auth over plain HTTP with
the publishable key, stores the access/refresh pair in cookies, and verifies
the access token locally against the project's public JWKS (ES256). Lives
outside src/ because it imports Flask.
"""
import os
from functools import wraps
from urllib.parse import urlsplit

import jwt
import requests
from flask import Blueprint, after_this_request, g, jsonify, request

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or ""
# Off for local http://127.0.0.1:5000; must be on in production (HTTPS).
COOKIE_SECURE = (os.environ.get("AUTH_COOKIE_SECURE") or "").lower() in ("1", "true", "yes", "on")
# Behind a proxy the request's own host isn't the public one, so it can be pinned.
APP_ORIGINS = {o.strip().rstrip("/") for o in
               (os.environ.get("APP_ORIGIN") or "").split(",") if o.strip()}

AUDIENCE = "authenticated"
# Pinned so a token can't pick its own algorithm (e.g. HS256 keyed with the public key).
ALGORITHMS = ["ES256"]
JWKS_CACHE_S = 600
HTTP_TIMEOUT_S = 10
# The cookie only has to outlive the refresh token; the access token's own
# `exp` is what bounds a session.
COOKIE_MAX_AGE = 30 * 24 * 3600

# __Host- makes the browser refuse the cookie unless it is Secure, host-only and
# Path=/, which blocks a sibling subdomain from planting one. Needs HTTPS.
_prefix = "__Host-" if COOKIE_SECURE else ""
ACCESS_COOKIE = _prefix + "sb-access"
REFRESH_COOKIE = _prefix + "sb-refresh"

_jwks = None
if SUPABASE_URL:
    # Caches the key set; an unknown `kid` (after a key rotation) triggers a refetch.
    _jwks = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
                            cache_jwk_set=True, lifespan=JWKS_CACHE_S)


def auth_configured():
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)


class AuthError(Exception):
    """A failure already phrased for the user, with the HTTP status to send."""
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


# ── Supabase Auth over HTTP ──────────────────────────────────────────────────
def _supabase(path, body=None, token=None):
    headers = {"apikey": SUPABASE_PUBLISHABLE_KEY}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        res = requests.post(f"{SUPABASE_URL}/auth/v1{path}", json=body or {},
                            headers=headers, timeout=HTTP_TIMEOUT_S)
    except requests.RequestException:
        raise AuthError("Could not reach the login server. Try again shortly.", 503)
    try:
        data = res.json() if res.content else {}
    except ValueError:
        data = {}
    if not res.ok:
        message = (data.get("msg") or data.get("error_description")
                   or data.get("message") or "Login server error.")
        # Pass on the client errors (bad password, rate limit); hide the rest.
        raise AuthError(message, res.status_code if res.status_code < 500 else 502)
    return data


def _refresh(refresh_token):
    return _supabase("/token?grant_type=refresh_token", {"refresh_token": refresh_token})


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


# ── Cookies ──────────────────────────────────────────────────────────────────
def _cookie_opts():
    return {"path": "/", "secure": COOKIE_SECURE, "httponly": True, "samesite": "Lax"}


def _set_session_cookies(pair):
    """Write the token pair onto whatever response this request ends up sending."""
    @after_this_request
    def _set(response):
        response.set_cookie(ACCESS_COOKIE, pair["access_token"],
                            max_age=COOKIE_MAX_AGE, **_cookie_opts())
        response.set_cookie(REFRESH_COOKIE, pair["refresh_token"],
                            max_age=COOKIE_MAX_AGE, **_cookie_opts())
        return response


def _clear_session_cookies():
    @after_this_request
    def _clear(response):
        response.delete_cookie(ACCESS_COOKIE, **_cookie_opts())
        response.delete_cookie(REFRESH_COOKIE, **_cookie_opts())
        return response


def _public_user(claims):
    return {"id": claims["sub"], "email": claims.get("email")}


def current_user():
    """Verified claims for the request's session cookies, or None if logged out.

    An expired access token is swapped for a new pair via the refresh token and
    the cookies are rewritten; if that is refused, the cookies are cleared.
    Raises AuthError(503) when Supabase can't be reached — an outage shouldn't
    log everyone out.
    """
    access = request.cookies.get(ACCESS_COOKIE)
    refresh = request.cookies.get(REFRESH_COOKIE)
    if not access and not refresh:
        return None

    if access:
        try:
            return verify_token(access)
        except jwt.PyJWKClientConnectionError:
            raise AuthError("Could not reach the auth server to verify the session.", 503)
        except jwt.ExpiredSignatureError:
            pass
        except jwt.PyJWTError:
            _clear_session_cookies()
            return None

    if not refresh:
        _clear_session_cookies()
        return None
    try:
        pair = _refresh(refresh)
        claims = verify_token(pair["access_token"])
    except AuthError as exc:
        if exc.status >= 500:
            raise
        _clear_session_cookies()
        return None
    except (jwt.PyJWTError, KeyError):
        _clear_session_cookies()
        return None
    _set_session_cookies(pair)
    return claims


def require_user(view):
    """Reject the request unless its session cookies hold a valid login.

    The verified claims are available to the view as `g.user`.
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        if _jwks is None:
            return jsonify({"error": "Login is not configured on this server."}), 503
        try:
            claims = current_user()
        except AuthError as exc:
            return jsonify({"error": str(exc)}), exc.status
        if claims is None:
            return jsonify({"error": "Please log in."}), 401
        g.user = claims
        return view(*args, **kwargs)
    return wrapper


# ── CSRF: Origin check on every state-changing request ───────────────────────
def _request_origin():
    source = request.headers.get("Origin")
    if source:
        return source
    referer = request.headers.get("Referer")
    if referer:
        parts = urlsplit(referer)
        return f"{parts.scheme}://{parts.netloc}"
    return None


def check_origin():
    """before_request hook: refuse cross-site POST/PUT/PATCH/DELETE.

    Browsers always send Origin (or at least Referer) on these; a request with
    neither comes from a non-browser client, which CSRF can't involve.
    """
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    source = _request_origin()
    if source is None:
        return None
    allowed = APP_ORIGINS or {request.host_url.rstrip("/")}
    if source.rstrip("/") not in allowed:
        return jsonify({"error": "Cross-site request refused."}), 403
    return None


# ── /auth routes ─────────────────────────────────────────────────────────────
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.before_request
def _auth_guard():
    if not auth_configured():
        return jsonify({"error": "Login is not configured on this server."}), 503
    # A cross-site HTML form can't send JSON without a CORS preflight, which
    # this app never grants.
    if request.method == "POST" and not request.is_json:
        return jsonify({"error": "Expected a JSON body."}), 415
    return None


@auth_bp.after_request
def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def _credentials():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip()
    password = str(data.get("password") or "")
    if not email or not password:
        raise AuthError("Enter your email and password.", 400)
    return email, password


@auth_bp.post("/signup")
def signup():
    try:
        email, password = _credentials()
        data = _supabase("/signup", {"email": email, "password": password})
    except AuthError as exc:
        return jsonify({"error": str(exc)}), exc.status
    # With email confirmation on, Supabase returns the user but no session.
    if "access_token" not in data:
        return jsonify({"user": None, "confirmation_required": True})
    _set_session_cookies(data)
    user = data.get("user") or {}
    return jsonify({"user": {"id": user.get("id"), "email": user.get("email")},
                    "confirmation_required": False})


@auth_bp.post("/login")
def login():
    try:
        email, password = _credentials()
        data = _supabase("/token?grant_type=password",
                         {"email": email, "password": password})
    except AuthError as exc:
        return jsonify({"error": str(exc)}), exc.status
    _set_session_cookies(data)
    user = data.get("user") or {}
    return jsonify({"user": {"id": user.get("id"), "email": user.get("email")}})


@auth_bp.post("/logout")
def logout():
    """End the session in Supabase, then clear the cookies whatever happens."""
    access = request.cookies.get(ACCESS_COOKIE)
    refresh = request.cookies.get(REFRESH_COOKIE)
    try:
        try:
            if not access:
                raise jwt.ExpiredSignatureError
            verify_token(access)
        except jwt.PyJWTError:
            # Supabase only accepts a live access token for logout.
            access = _refresh(refresh)["access_token"] if refresh else None
        if access:
            _supabase("/logout?scope=local", token=access)
    except (AuthError, KeyError):
        pass    # the session is unusable or already gone; still drop the cookies
    _clear_session_cookies()
    return jsonify({"user": None})


@auth_bp.get("/session")
def session():
    """Who is logged in, for the page header; {user: null} rather than a 401."""
    try:
        claims = current_user()
    except AuthError as exc:
        return jsonify({"user": None, "error": str(exc)}), exc.status
    return jsonify({"user": _public_user(claims) if claims else None})
