from __future__ import annotations

import re

from core import api_client
from core.api_client import ApiError, AuthResult

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 8
GENERIC_LOGIN_ERROR = "Wrong email or password"
SIGNUP_FAILED = (
    "Couldn't create the account. The email may already be registered, "
    "or the password may be too weak."
)


class AuthError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfirmationRequired(AuthError):
    pass


def validate_email(email: str) -> str | None:
    if not EMAIL_RE.fullmatch((email or "").strip()):
        return "Enter a valid email address."
    return None


def validate_password(password: str) -> str | None:
    if not password or len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    return None


def public_user(result: AuthResult) -> dict[str, str | None]:
    email = result.email
    display = email.split("@")[0] if email else "there"
    return {
        "id": result.user_id,
        "email": email,
        "username": email,
        "display_name": display,
        "access_token": result.access_token,
        "refresh_token": result.refresh_token,
    }


def signup(email: str, password: str, confirm_password: str) -> dict[str, str | None]:
    email = (email or "").strip()
    error = validate_email(email)
    if error:
        raise AuthError(error)
    error = validate_password(password)
    if error:
        raise AuthError(error)
    if password != confirm_password:
        raise AuthError("Passwords do not match.")
    try:
        result = api_client.signup(email, password)
    except ApiError as exc:
        # The backend's 400 text contains internal exception messages: never show it.
        if exc.status_code == 400:
            raise AuthError(SIGNUP_FAILED) from exc
        raise AuthError(exc.message) from exc
    if result.confirmation_required or not result.access_token:
        raise ConfirmationRequired(
            "Account created, but this server still requires email "
            "confirmation, so you can't log in yet."
        )
    if not result.user_id:
        raise AuthError("Sign-up did not return a user. Try logging in.")
    return public_user(result)


def login(email: str, password: str) -> dict[str, str | None]:
    email = (email or "").strip()
    if not email or not password:
        raise AuthError(GENERIC_LOGIN_ERROR)
    try:
        result = api_client.login(email, password)
    except ApiError as exc:
        if exc.status_code in {400, 401, 403}:
            raise AuthError(GENERIC_LOGIN_ERROR) from exc
        raise AuthError(exc.message) from exc
    if not result.user_id or not result.access_token:
        raise AuthError(GENERIC_LOGIN_ERROR)
    return public_user(result)
