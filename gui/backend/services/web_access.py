from __future__ import annotations

import hashlib
import hmac
import secrets
from ipaddress import ip_address

from fastapi import HTTPException, Request, Response

from core.config import (
    WEB_SESSION_COOKIE_SECURE,
    WEB_SESSION_SECRET,
    WEB_TRUST_PROXY_HEADERS,
)
from core.policy import is_web_mode
from models.job import Job


SESSION_COOKIE_NAME = "ligq2_web_session"


def _digest(value: str) -> str:
    return hmac.new(
        WEB_SESSION_SECRET.encode("utf-8"),
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def prepare_web_session(request: Request) -> str | None:
    if not is_web_mode():
        return None

    token = request.cookies.get(SESSION_COOKIE_NAME)
    is_new = not token or len(token) < 32
    if is_new:
        token = secrets.token_urlsafe(32)
    request.state.session_hash = _digest(token)
    return token if is_new else None


def set_web_session_cookie(response: Response, token: str | None) -> None:
    if token is None:
        return
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=WEB_SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def session_hash(request: Request) -> str | None:
    if not is_web_mode():
        return None
    value = getattr(request.state, "session_hash", None)
    if value:
        return str(value)
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return _digest(token) if token else None


def require_job_access(request: Request, job: Job | None) -> Job:
    if job is None or (
        is_web_mode()
        and (
            not job.owner_session_hash
            or job.owner_session_hash != session_hash(request)
        )
    ):
        raise HTTPException(
            404,
            detail={
                "error": "job_not_found",
                "message": "The requested job was not found.",
                "details": None,
            },
        )
    return job


def require_resource_management() -> None:
    if is_web_mode():
        raise HTTPException(
            403,
            detail={
                "error": "feature_disabled",
                "message": "Resource management is disabled on the public web service.",
                "details": None,
            },
        )


def client_ip_hash(request: Request) -> str:
    raw_ip = request.client.host if request.client else "unknown"
    if WEB_TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            raw_ip = forwarded.split(",", 1)[0].strip()
        elif request.headers.get("x-real-ip"):
            raw_ip = request.headers["x-real-ip"].strip()
    try:
        normalized = str(ip_address(raw_ip))
    except ValueError:
        normalized = "unknown"
    return _digest(f"ip:{normalized}")
