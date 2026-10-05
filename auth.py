"""Access control for the Flask back end: Supabase logins and guest mode.

Every /api/* request needs a token: either a Supabase session (access/refresh
pair in HttpOnly cookies) or a signed guest cookie. The browser never sees a
Supabase token: Flask calls Supabase Auth over plain HTTP with the publishable
key and verifies access tokens locally against the project's public JWKS
(ES256). Lives outside src/ because it imports Flask.
"""
import os
from functools import wraps
from urllib.parse import urlsplit

import jwt
import requests
from flask import (Blueprint, after_this_request, current_app, g, jsonify,
                   make_response, request)
from itsdangerous import BadSignature, URLSafeTimedSerializer

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
GUEST_LIMIT = 3

# __Host- makes the browser refuse the cookie unless it is Secure, host-only and
# Path=/, which blocks a sibling subdomain from planting one. Needs HTTPS.
_prefix = "__Host-" if COOKIE_SECURE else ""
ACCESS_COOKIE = _prefix + "sb-access"
REFRESH_COOKIE = _prefix + "sb-refresh"
GUEST_COOKIE = _prefix + "fl-guest"

_jwks = None
if SUPABASE_URL:
    # Caches the key set; an unknown `kid` (after a key rotation) triggers a refetch.
    _jwks = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
                            cache_jwk_set=True, lifespan=JWKS_CACHE_S)


def auth_configured():
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)


def guest_configured():
    return bool(current_app.secret_key)


class AuthError(Exception):
    """A failure already phrased for the user, with the HTTP status to send."""
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def _deny(status, code, message):
    return jsonify({"error": message, "code": code}), status


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


# ── Guest cookie ─────────────────────────────────────────────────────────────
# Signed with the app's SECRET_KEY, so the count can't be edited. Deleting the
# cookie does reset it — a known limit of browser-side counting.
def _guest_signer():
    return URLSafeTimedSerializer(current_app.secret_key, salt="foodlens-guest")


def guest_count():
    """Analyses this guest has used, or None without a valid guest cookie."""
    raw = request.cookies.get(GUEST_COOKIE)
    if not raw or not guest_configured():
        return None
    try:
        data = _guest_signer().loads(raw, max_age=COOKIE_MAX_AGE)
        return max(0, int(data["n"]))
    except (BadSignature, KeyError, TypeError, ValueError):
        return None


def _set_guest_cookie(response, count):
    response.set_cookie(GUEST_COOKIE, _guest_signer().dumps({"n": count}),
                        max_age=COOKIE_MAX_AGE, **_cookie_opts())


def _guest_info(count):
    return {"remaining": max(0, GUEST_LIMIT - count), "limit": GUEST_LIMIT}


# ── Who is asking ────────────────────────────────────────────────────────────
def current_user():
    """Verified claims for the request's session cookies, or None if logged out.

    An expired access token is swapped for a new pair via the refresh token and
    the cookies are rewritten; if that is refused, the cookies are cleared.
    Raises AuthError(503) when Supabase can't be reached — an outage shouldn't
    log everyone out. Resolved once per request.
    """
    if "auth_user" in g:
        return g.auth_user
    g.auth_user = _resolve_user() if _jwks is not None else None
    return g.auth_user


def _resolve_user():
    access = request.cookies.get(ACCESS_COOKIE)
    refresh = request.cookies.get(REFRESH_COOKIE)
    if not access and not refresh:
        return None

    if access:
        try:
            claims = verify_token(access)
            g.access_token = access
            return claims
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
    g.access_token = pair["access_token"]
    return claims


def is_guest():
    """True when the request is running on a guest cookie rather than a login."""
    try:
        return current_user() is None
    except AuthError:
        return True


def has_access():
    """Whether the visitor may see the app: a login, or a guest cookie."""
    try:
        if current_user():
            return True
    except AuthError:
        pass
    return guest_count() is not None


def require_token():
    """before_request hook: every /api/* call needs a login or a guest cookie."""
    if not request.path.startswith("/api/"):
        return None
    try:
        if current_user():
            return None
    except AuthError as exc:
        return _deny(exc.status, "auth_unavailable", str(exc))
    if guest_count() is not None:
        return None
    return _deny(401, "auth_required", "Log in or continue as a guest first.")


def require_user(view):
    """Reject the request unless its session cookies hold a valid login.

    The verified claims are available to the view as `g.user`, and the access
    token they came from as `g.access_token` (for calls made as the user).
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        if _jwks is None:
            return _deny(503, "auth_unavailable", "Login is not configured on this server.")
        try:
            claims = current_user()
        except AuthError as exc:
            return _deny(exc.status, "auth_unavailable", str(exc))
        if claims is None:
            return _deny(401, "login_required", "Log in to use this feature.")
        g.user = claims
        return view(*args, **kwargs)
    return wrapper


def guest_quota(view):
    """Logged-in users are unlimited; guests get GUEST_LIMIT successful runs."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not is_guest():
            return view(*args, **kwargs)
        used = guest_count()
        if used is None:
            return _deny(401, "auth_required", "Log in or continue as a guest first.")
        if used >= GUEST_LIMIT:
            return _deny(401, "guest_limit_reached",
                         f"You've used your {GUEST_LIMIT} free analyses. "
                         f"Log in or sign up to keep going.")
        response = make_response(view(*args, **kwargs))
        if response.status_code == 200:
            _set_guest_cookie(response, used + 1)
        return response
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
def _json_only():
    # A cross-site HTML form can't send JSON without a CORS preflight, which
    # this app never grants.
    if request.method == "POST" and not request.is_json:
        return jsonify({"error": "Expected a JSON body."}), 415
    return None


@auth_bp.after_request
def _no_store(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def _login_unavailable():
    return jsonify({"error": "Login is not configured on this server."}), 503


def _credentials():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip()
    password = str(data.get("password") or "")
    if not email or not password:
        raise AuthError("Enter your email and password.", 400)
    return email, password


@auth_bp.post("/signup")
def signup():
    if not auth_configured():
        return _login_unavailable()
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
    if not auth_configured():
        return _login_unavailable()
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
    """End the session in Supabase, then clear the cookies whatever happens.

    The guest cookie is left alone, so logging out never hands back a fresh
    set of free analyses.
    """
    access = request.cookies.get(ACCESS_COOKIE)
    refresh = request.cookies.get(REFRESH_COOKIE)
    if auth_configured():
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


@auth_bp.post("/guest")
def guest():
    """Start guest mode. An existing guest keeps their count rather than resetting."""
    if not guest_configured():
        return jsonify({"error": "Guest mode is not configured on this server."}), 503
    used = guest_count()
    response = make_response(jsonify({"guest": _guest_info(used or 0)}))
    if used is None:
        _set_guest_cookie(response, 0)
    return response


@auth_bp.get("/session")
def session():
    """Who is visiting, for the page header; nulls rather than a 401."""
    try:
        claims = current_user()
    except AuthError as exc:
        return jsonify({"user": None, "guest": None, "error": str(exc)}), exc.status
    if claims:
        return jsonify({"user": _public_user(claims), "guest": None})
    used = guest_count()
    return jsonify({"user": None,
                    "guest": _guest_info(used) if used is not None else None})
