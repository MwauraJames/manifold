from __future__ import annotations

import uuid

import streamlit as st

ALL_DOCS = "__all__"  # scope value meaning "every document in the subject"


def _defaults() -> dict:
    return {
        "user": None,
        "subject": None,
        "view": "Chat",
        "show_new_subject": False,
        "show_delete_subject": False,
        "chat_id": None,        # None until the first /ask creates the chat
        "chat_doc_id": None,    # scope of an existing chat (None = all documents)
        "chat_subject": None,   # subject the chat state belongs to
        "chat_messages": [],
        "uploader_version": 0,
        "chat_view_id": uuid.uuid4().hex,  # changes whenever the chat screen is swapped for another chat
        "editing_last": False,
        "stream_job": None,                # background answer being generated (core/stream_job.py)
    }


def init_state() -> None:
    for key, value in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_view() -> None:
    """Mark the chat screen as a different conversation (see StreamJob.view_id)."""
    st.session_state["chat_view_id"] = uuid.uuid4().hex
    st.session_state["editing_last"] = False


def reset_chat() -> None:
    """Back to a blank 'new chat' screen."""
    new_view()
    st.session_state["chat_id"] = None
    st.session_state["chat_doc_id"] = None
    st.session_state["chat_messages"] = []
    st.session_state["scope_doc"] = ALL_DOCS


def reset_all() -> None:
    """Log out: wipe everything for this browser session and forget the saved login."""
    st.session_state.clear()
    init_state()
    # Never auto-login again in this browser session (the cookie the server saw when
    # the page loaded is still in st.context), and delete the saved login.
    st.session_state["_restore_attempted"] = True
    st.session_state["_session_clear"] = True
