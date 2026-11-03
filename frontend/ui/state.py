from __future__ import annotations

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
    }


def init_state() -> None:
    for key, value in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    """Back to a blank 'new chat' screen."""
    st.session_state["chat_id"] = None
    st.session_state["chat_doc_id"] = None
    st.session_state["chat_messages"] = []
    st.session_state["scope_doc"] = ALL_DOCS


def reset_all() -> None:
    """Log out: wipe everything for this browser session."""
    st.session_state.clear()
    init_state()
