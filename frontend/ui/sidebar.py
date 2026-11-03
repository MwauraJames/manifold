from __future__ import annotations

from collections import Counter

import streamlit as st

from core import api_client
from core.api_client import ApiError, ChatSummary, DocumentRecord, ModelList, SubjectRecord
from core.utils import validate_subject_name
from ui import state


# ------------------------------------------------------------------ callbacks
# Callbacks run before the script reruns, so they can safely change widget state.


def _select_subject(name: str) -> None:
    st.session_state["subject"] = name
    st.session_state["chat_subject"] = name
    st.session_state["view"] = "Chat"
    state.reset_chat()


def _select_chat(chat: ChatSummary, subject: str) -> None:
    try:
        messages = api_client.get_chat_messages(chat.chat_id)
    except ApiError as exc:
        st.session_state["flash"] = exc.message
        return
    st.session_state["chat_id"] = chat.chat_id
    st.session_state["chat_doc_id"] = chat.doc_id
    st.session_state["chat_messages"] = messages
    st.session_state["chat_subject"] = subject
    st.session_state["view"] = "Chat"


def _delete_chat(chat_id: str) -> None:
    try:
        api_client.delete_chat(chat_id)
    except ApiError as exc:
        if exc.status_code != 404:
            st.session_state["flash"] = exc.message
            return
    if st.session_state.get("chat_id") == chat_id:
        state.reset_chat()
    st.session_state["flash"] = "Chat deleted."


def _logout() -> None:
    state.reset_all()


def prompt_delete_subject() -> None:
    st.session_state["show_delete_subject"] = True


# -------------------------------------------------------------------- dialogs


@st.dialog("New subject")
def new_subject_dialog(existing_names: list[str]) -> None:
    # Clear the flag first so the dialog doesn't reopen on the next full rerun.
    st.session_state["show_new_subject"] = False
    name = st.text_input("Name", max_chars=60, placeholder="e.g. Machine Learning")
    description = st.text_input("Description (optional)", max_chars=200)
    if st.button("Create", type="primary"):
        error = validate_subject_name(name, existing_names)
        if error:
            st.error(error)
            return
        try:
            record = api_client.create_subject(name.strip(), description.strip() or None)
        except ApiError as exc:
            st.error(exc.message)
            return
        st.session_state["subject"] = record.name
        st.session_state["chat_subject"] = record.name
        st.session_state["view"] = "Documents"  # next step is uploading notes
        state.reset_chat()
        st.rerun()


@st.dialog("Delete subject")
def delete_subject_dialog(subject_name: str) -> None:
    st.session_state["show_delete_subject"] = False
    try:
        docs = api_client.list_documents(subject_name)
        chats = api_client.list_chats(subject_name)
    except ApiError:
        docs, chats = [], []
    st.warning(
        f"This permanently deletes **{subject_name}**, its {len(docs)} document(s) "
        f"and {len(chats)} chat(s), including all indexed data. This can't be undone."
    )
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Cancel", use_container_width=True):
        st.rerun()
    if delete_col.button("Delete", type="primary", use_container_width=True):
        with st.spinner("Deleting…"):
            try:
                api_client.delete_subject(subject_name)
            except ApiError as exc:
                st.error(exc.message)
                return
        st.session_state["flash"] = f"Deleted “{subject_name}”."
        st.session_state["subject"] = None
        st.session_state["chat_subject"] = None
        state.reset_chat()
        st.rerun()


# --------------------------------------------------------------------- models


def _load_models() -> ModelList:
    """Fetch the server's model list once per browser session."""
    cached = st.session_state.get("_models")
    if cached is not None:
        return cached
    try:
        models = api_client.list_models()
    except ApiError:
        models = ModelList()
    st.session_state["_models"] = models
    return models


def _render_model_picker() -> None:
    models = _load_models()
    if not models.models:
        st.session_state["model_id"] = None
        return
    ids = [m.id for m in models.models]
    labels = {m.id: f"{m.name} · {m.provider}" if m.provider else m.name for m in models.models}
    if st.session_state.get("model_id") not in ids:
        st.session_state["model_id"] = models.default_model if models.default_model in ids else ids[0]
    st.selectbox("Model", ids, format_func=lambda i: labels[i], key="model_id")


# -------------------------------------------------------------------- sidebar


def render_sidebar(
    user: dict,
    subjects: list[SubjectRecord],
    documents: list[DocumentRecord],
) -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash)

    names = [s.name for s in subjects]
    current = st.session_state.get("subject")
    ready_counts = Counter(d.subject for d in documents if d.status == "ready")

    with st.sidebar:
        st.markdown("### Study Assistant")
        if st.button("＋ New subject", use_container_width=True):
            st.session_state["show_new_subject"] = True

        st.caption("SUBJECTS")
        if not subjects:
            st.caption("No subjects yet.")
        for index, subject in enumerate(subjects):
            count = ready_counts.get(subject.name, 0)
            label = f"{subject.name}  ·  {count}" if count else subject.name
            st.button(
                label,
                key=f"subject_btn_{index}",
                type="primary" if subject.name == current else "secondary",
                use_container_width=True,
                on_click=_select_subject,
                args=(subject.name,),
            )

        if current:
            st.divider()
            st.button("New chat", icon="✏️", use_container_width=True, on_click=state.reset_chat)
            st.caption("CHATS")
            try:
                chats = api_client.list_chats(current)
            except ApiError as exc:
                chats = []
                st.caption(exc.message)
            chats = sorted(chats, key=lambda c: str(c.updated_at or ""), reverse=True)
            if not chats:
                st.caption("No chats yet in this subject.")
            for chat in chats[:30]:
                title = chat.title if len(chat.title) <= 30 else chat.title[:29] + "…"
                row_open, row_menu = st.columns([5, 1])
                row_open.button(
                    title,
                    key=f"chat_btn_{chat.chat_id}",
                    type="primary" if chat.chat_id == st.session_state.get("chat_id") else "secondary",
                    use_container_width=True,
                    on_click=_select_chat,
                    args=(chat, current),
                )
                with row_menu.popover("⋯"):
                    st.button(
                        "Delete chat",
                        key=f"chat_del_{chat.chat_id}",
                        on_click=_delete_chat,
                        args=(chat.chat_id,),
                    )

            st.divider()
            _render_model_picker()

        st.divider()
        st.caption(user.get("email") or "")
        st.button("Log out", use_container_width=True, on_click=_logout)
        st.caption("Theme: open the ⋮ menu (top right) → Settings.")

    if st.session_state.get("show_new_subject"):
        new_subject_dialog(names)
