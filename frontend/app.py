from __future__ import annotations

import streamlit as st
from streamlit.runtime.scriptrunner import RerunException

from core import api_client, session_store
from core.api_client import ApiError
from core.config import ENABLE_DELETE
from ui import state
from ui.auth_ui import render_auth
from ui.chat_ui import render_chat
from ui.documents_ui import render_documents
from ui.sidebar import delete_subject_dialog, prompt_delete_subject, render_sidebar

# Theme: set in .streamlit/config.toml (dark by default). Users can switch to
# light from Streamlit's own menu (top right -> Settings), so no custom CSS here.
st.set_page_config(page_title="Manifold", page_icon=":material/hub:", layout="wide")
state.init_state()

# After a browser refresh the session is empty: log back in from the saved cookie.
session_store.restore()


def page() -> None:
    user = st.session_state.get("user")
    if not user:
        render_auth()
        return

    try:
        subjects = api_client.list_subjects()
        documents = api_client.list_documents()
    except ApiError as exc:
        if exc.status_code == 401:
            # Token expired and the refresh failed: back to the login screen.
            state.reset_all()
            st.session_state["auth_notice"] = "Your session expired. Please log in again."
            st.rerun()
        render_sidebar(user, [], [])
        st.error(exc.message)
        return

    # Make sure the selected subject still exists (it may have been deleted).
    names = [s.name for s in subjects]
    subject_name = st.session_state.get("subject")
    if subject_name not in names:
        subject_name = names[0] if names else None
        st.session_state["subject"] = subject_name

    render_sidebar(user, subjects, documents)

    if not subject_name:
        st.title("Manifold")
        st.info("Create a subject to upload notes and start chatting. Each subject stays private to you.")
        if st.button("New subject", key="empty_new_subject"):
            st.session_state["show_new_subject"] = True
            st.rerun()
        return

    subject_docs = [d for d in documents if d.subject == subject_name]

    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.title(subject_name)
    with header_right:
        if ENABLE_DELETE and st.button("Delete subject"):
            prompt_delete_subject()

    if st.session_state.get("show_delete_subject"):
        delete_subject_dialog(subject_name)

    # A radio instead of st.tabs: st.chat_input only pins to the bottom of the page
    # when it is NOT inside a tab or container.
    view = st.radio(
        "View",
        ["Chat", "Documents"],
        horizontal=True,
        label_visibility="collapsed",
        key="view",
    )
    if view == "Chat":
        render_chat(subject_name, subject_docs)
    else:
        render_documents(subject_name, subject_docs)


# Save (or delete) the login cookie when the run ends. page() returns instead of
# calling st.stop(): after st.stop() Streamlit refuses any further st.* call, so
# nothing could be rendered. A run that ends in st.rerun() skips the flush; the next
# run does it once the page is stable.
try:
    page()
except RerunException:
    raise
else:
    session_store.flush()
