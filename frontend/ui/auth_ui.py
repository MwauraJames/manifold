from __future__ import annotations

import streamlit as st

from core import auth, session_store
from core.auth import AuthError, ConfirmationRequired
from ui import state

# Only ONE form is rendered at a time (no tabs). With tabs, every login and sign-up
# field sits in the page at once, and browsers can't tell which password field is
# which, so they stop offering to fill, suggest or save passwords.
#
# The autocomplete values tell the browser what each field is:
#   username          the email used as the account name
#   current-password  login: fill the saved password
#   new-password      sign-up / reset: suggest a strong password and offer to save it
#   one-time-code     the emailed reset code

TAGLINE = (
    "**Ask your documents anything.** Built for students and researchers: turn lecture notes, "
    "papers and slides into answers. Just as useful for reports, policies, runbooks and project docs."
)


def _go(view: str) -> None:
    st.session_state["auth_view"] = view


def _finish(user: dict, message: str | None = None) -> None:
    state.init_state()
    st.session_state["user"] = user
    st.session_state["_session_dirty"] = True    # save the login in the browser
    st.session_state["_session_clear"] = False
    st.session_state["_store_credentials"] = True  # offer to save the password
    st.session_state["auth_view"] = "login"
    st.session_state.pop("reset_email", None)
    if message:
        st.session_state["flash"] = message
    state.reset_chat()
    st.session_state["subject"] = None
    st.rerun()


def render_auth() -> None:
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("Manifold")
        st.markdown(TAGLINE)

        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.info(notice)

        view = st.session_state.get("auth_view", "login")
        if view == "signup":
            _signup_view()
        elif view == "forgot":
            _forgot_view()
        elif view == "reset" and st.session_state.get("reset_email"):
            _reset_view()
        else:
            _login_view()

    session_store.install_credential_capture()


# ------------------------------------------------------------------- views


def _login_view() -> None:
    st.subheader("Log in")
    with st.form("login_form", border=False):
        email = st.text_input("Email", key="login_email", autocomplete="username")
        password = st.text_input(
            "Password", type="password", key="login_password", autocomplete="current-password"
        )
        submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
    if submitted:
        try:
            user = auth.login(email, password)
        except AuthError as exc:
            st.error(exc.message)
        else:
            _finish(user)

    left, right = st.columns(2)
    left.button("Forgot password?", type="tertiary", on_click=_go, args=("forgot",), key="to_forgot")
    right.button("Create an account", type="tertiary", on_click=_go, args=("signup",), key="to_signup")


def _signup_view() -> None:
    st.subheader("Create your account")
    with st.form("signup_form", border=False):
        email = st.text_input("Email", key="signup_email", autocomplete="username")
        password = st.text_input(
            "Password",
            type="password",
            key="signup_password",
            autocomplete="new-password",
            help=f"At least {auth.MIN_PASSWORD_LEN} characters. Your browser can suggest a strong one.",
        )
        confirm = st.text_input(
            "Confirm password", type="password", key="signup_confirm", autocomplete="new-password"
        )
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if submitted:
        try:
            user = auth.signup(email, password, confirm)
        except ConfirmationRequired as exc:
            st.warning(exc.message)
        except AuthError as exc:
            st.error(exc.message)
        else:
            _finish(user)
    st.button("Already have an account? Log in", type="tertiary", on_click=_go, args=("login",), key="to_login")


def _forgot_view() -> None:
    st.subheader("Reset your password")
    st.caption("Enter your email and we'll send you a code to set a new password.")
    with st.form("forgot_form", border=False):
        email = st.text_input("Email", key="forgot_email", autocomplete="username")
        submitted = st.form_submit_button("Send reset code", type="primary", use_container_width=True)
    if submitted:
        try:
            cleaned = auth.request_reset(email)
        except AuthError as exc:
            st.error(exc.message)
        else:
            st.session_state["reset_email"] = cleaned
            st.session_state["auth_view"] = "reset"
            st.rerun()
    st.button("Back to log in", type="tertiary", on_click=_go, args=("login",), key="forgot_back")


def _resend_code(email: str) -> None:
    try:
        auth.request_reset(email)
    except AuthError as exc:
        st.session_state["reset_message"] = ("error", exc.message)
    else:
        st.session_state["reset_message"] = ("success", "We sent a new code.")


def _reset_view() -> None:
    email = st.session_state["reset_email"]
    st.subheader("Check your email")
    st.caption(
        f"If an account exists for **{email}**, we've sent a code. "
        "Enter it below with your new password."
    )
    kind, text = st.session_state.pop("reset_message", (None, None))
    if kind == "success":
        st.success(text)
    elif kind == "error":
        st.error(text)

    with st.form("reset_form", border=False):
        # Shown read-only so the browser knows which account this new password belongs to.
        st.text_input("Email", value=email, disabled=True, autocomplete="username", key="reset_email_shown")
        code = st.text_input("Code from the email", key="reset_code", autocomplete="one-time-code", max_chars=12)
        password = st.text_input(
            "New password", type="password", key="reset_password", autocomplete="new-password"
        )
        confirm = st.text_input(
            "Confirm new password", type="password", key="reset_confirm", autocomplete="new-password"
        )
        submitted = st.form_submit_button("Set new password", type="primary", use_container_width=True)
    if submitted:
        try:
            user = auth.reset_password(email, code, password, confirm)
        except AuthError as exc:
            st.error(exc.message)
        else:
            _finish(user, "Password updated. You're logged in.")

    left, right = st.columns(2)
    left.button("Send a new code", type="tertiary", on_click=_resend_code, args=(email,), key="reset_resend")
    right.button("Back to log in", type="tertiary", on_click=_go, args=("login",), key="reset_back")
