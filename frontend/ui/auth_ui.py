from __future__ import annotations

import streamlit as st

from core import auth
from core.auth import AuthError, ConfirmationRequired
from ui import state


def _finish(user: dict) -> None:
    state.init_state()
    st.session_state["user"] = user
    st.session_state["_session_dirty"] = True   # save the login in the browser
    st.session_state["_session_clear"] = False
    state.reset_chat()
    st.session_state["subject"] = None
    st.rerun()


def render_auth() -> None:
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("Manifold")
        st.caption("Upload your notes, organise them by subject, and ask questions about them.")

        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.info(notice)

        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                try:
                    user = auth.login(email, password)
                except AuthError as exc:
                    st.error(exc.message)
                else:
                    _finish(user)

        with signup_tab:
            with st.form("signup_form"):
                email = st.text_input("Email", key="signup_email")
                password = st.text_input(
                    "Password", type="password", key="signup_password",
                    help=f"At least {auth.MIN_PASSWORD_LEN} characters.",
                )
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
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
