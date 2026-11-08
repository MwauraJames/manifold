from __future__ import annotations

import json
from urllib.parse import quote, unquote
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from core import api_client, auth
from core.api_client import ApiError
from core.config import SESSION_DAYS

# Keeps a user logged in across browser refreshes.
#
# A Streamlit page reload creates a brand-new session, so everything in
# st.session_state is lost. To survive that we save the *refresh token* (not the
# access token) and the email in a browser cookie, and on the next page load we
# read it back and trade it for a fresh session.
#
#   write: a tiny script in a zero-height component sets the cookie
#   read:  st.context.cookies (what the browser sent when the page loaded)
#
# The cookie can't be HttpOnly because JavaScript sets it, so treat it like any
# stored login: it lives for SESSION_DAYS, is cleared on logout, and the server
# can revoke it.

COOKIE_NAME = "manifold_session"
_DIRTY = "_session_dirty"        # tokens changed: rewrite the cookie
_CLEAR = "_session_clear"        # logged out or token rejected: delete the cookie
_ATTEMPTED = "_restore_attempted"
_STORE = "_store_credentials"  # a login/sign-up just succeeded: offer to save it


# ------------------------------------------------------------------- reading


def _read_cookie() -> str | None:
    try:
        value = st.context.cookies.get(COOKIE_NAME)
    except Exception:  # older Streamlit, or no browser attached
        return None
    return value if isinstance(value, str) else None


def read_saved() -> dict | None:
    raw = _read_cookie()
    if not raw:
        return None
    try:
        data = json.loads(unquote(raw).strip('"'))
    except ValueError:
        return None
    if isinstance(data, dict) and data.get("refresh_token"):
        return data
    return None


def restore() -> None:
    """Log the user back in from the saved cookie. Runs once per browser session."""
    if st.session_state.get(_ATTEMPTED):
        return
    st.session_state[_ATTEMPTED] = True
    if st.session_state.get("user"):
        return
    saved = read_saved()
    if not saved:
        return
    try:
        result = api_client.refresh_with(saved["refresh_token"])
    except ApiError:
        return  # server unreachable: show the login page but keep the cookie
    if result is None:
        st.session_state[_CLEAR] = True  # the token was rejected: forget it
        return
    result.email = saved.get("email") or ""
    st.session_state["user"] = auth.public_user(result)
    st.session_state[_DIRTY] = True  # the refresh token just rotated


# ------------------------------------------------------------------- writing


def _render(html: str) -> None:
    """Render invisible HTML that is allowed to run JavaScript on the app's origin."""
    if hasattr(st, "iframe"):  # Streamlit >= 1.5x: st.components.v1.html is deprecated
        st.iframe(html.strip(), height=1)
    else:
        components.html(html, height=0)


def _inject(payload: dict | None) -> None:
    if payload is None:
        value, max_age = "", 0
    else:
        value, max_age = quote(json.dumps(payload), safe=""), SESSION_DAYS * 86400
    script = f"""
    <script>
      (function () {{
        try {{
          var secure = window.parent.location.protocol === "https:" ? "; Secure" : "";
          window.parent.document.cookie =
            "{COOKIE_NAME}={value}; path=/; max-age={max_age}; SameSite=Lax" + secure;
        }} catch (e) {{}}
      }})();
    </script>
    <!-- {uuid4().hex} -->
    """
    _render(script)


def flush() -> None:
    """Write or delete the saved-login cookie if something changed. Call at the end of a run."""
    if st.session_state.get(_CLEAR):
        st.session_state[_CLEAR] = False
        st.session_state[_DIRTY] = False
        _inject(None)
    elif st.session_state.get(_DIRTY):
        st.session_state[_DIRTY] = False
        user = st.session_state.get("user")
        if isinstance(user, dict) and user.get("refresh_token"):
            _inject({"refresh_token": user["refresh_token"], "email": user.get("email") or ""})

    if st.session_state.get(_STORE):
        st.session_state[_STORE] = False
        _render(_STORE_JS)


# ------------------------------------------------- browser password manager
#
# Streamlit renders no real <form> element, so browsers can't tell when a sign-up
# or login succeeded and often don't offer to save the password. Chrome and Edge
# support the Credential Management API, which lets a page hand over a credential
# explicitly. The capture script remembers what was typed when a form is
# submitted (in the browser only, never sent to the server); once the server
# confirms success, the store script gives it to the browser. Other browsers
# ignore both scripts and use their own detection.

_CAPTURE_JS = """
<script>
  (function () {
    try {
      var w = window.parent, d = w.document;
      if (w.__manifoldCapture) return;
      w.__manifoldCapture = true;
      function grab() {
        var u = d.querySelector('input[autocomplete="username"]');
        var p = d.querySelector('input[autocomplete="new-password"], input[autocomplete="current-password"]');
        if (u && p && u.value && p.value) w.__manifoldPending = { id: u.value, password: p.value };
      }
      d.addEventListener("click", function (e) {
        if (e.target.closest && e.target.closest('[data-testid="stFormSubmitButton"]')) grab();
      }, true);
      d.addEventListener("keydown", function (e) { if (e.key === "Enter") grab(); }, true);
    } catch (e) {}
  })();
</script>
"""

_STORE_JS = """
<script>
  (function () {
    try {
      var w = window.parent, c = w.__manifoldPending;
      if (c && c.id && c.password && w.PasswordCredential && w.navigator.credentials) {
        w.navigator.credentials.store(new w.PasswordCredential({ id: c.id, password: c.password, name: c.id }));
      }
      w.__manifoldPending = null;
    } catch (e) {}
  })();
</script>
"""


def install_credential_capture() -> None:
    """Render on the login/sign-up pages so the browser can be asked to save the password."""
    _render(_CAPTURE_JS)
