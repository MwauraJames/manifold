from __future__ import annotations

import streamlit as st
from streamlit.runtime.scriptrunner import RerunException

from core import api_client, session_store
from core.api_client import ApiError
from ui import state, styles
from ui.auth_ui import render_auth
from ui.chat_ui import render_chat
from ui.documents_ui import render_documents_panel
from ui.sidebar import delete_subject_dialog, render_sidebar
from ui.topbar import render_topbar

# Theme: set in .streamlit/config.toml (dark by default). Users can switch to
# light from Streamlit's own menu (top right -> Settings), so no custom colours here.
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
        styles.inject(False)
        render_sidebar(user, [], [])
        st.error(exc.message)
        return

    # Make sure the selected subject still exists (it may have been deleted).
    names = [s.name for s in subjects]
    subject_name = st.session_state.get("subject")
    if subject_name not in names:
        subject_name = names[0] if names else None
        st.session_state["subject"] = subject_name

    subject_docs = [d for d in documents if d.subject == subject_name] if subject_name else []
    docs_open = bool(subject_name) and state.docs_panel_open(subject_docs)
    styles.inject(docs_open)
    render_sidebar(user, subjects, documents)

    if not subject_name:
        with st.container(key="hero"):
            st.markdown(
                '<div class="hero-title">Welcome to Manifold</div>'
                '<div class="hero-sub">Create a subject, add your notes, then ask questions about them.</div>',
                unsafe_allow_html=True,
            )
            _, middle, _ = st.columns([1, 2, 1])
            if middle.button("Create your first subject", type="primary", use_container_width=True, key="empty_new_subject"):
                st.session_state["show_new_subject"] = True
                st.rerun()
        return

    if st.session_state.get("show_delete_subject"):
        delete_subject_dialog(subject_name)

    render_topbar(subject_name, subject_docs, docs_open)
    if docs_open:
        render_documents_panel(subject_name, subject_docs)
    render_chat(subject_name, subject_docs)


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
