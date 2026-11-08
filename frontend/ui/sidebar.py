from __future__ import annotations

from collections import Counter

import streamlit as st

from core import api_client
from core.api_client import ApiError, ChatSummary, DocumentRecord, SubjectRecord
from core.utils import validate_subject_name
from ui import state


# ------------------------------------------------------------------ callbacks
# Callbacks run before the script reruns, so they can safely change widget state.


def _select_subject(name: str) -> None:
    st.session_state["subject"] = name
    st.session_state["chat_subject"] = name
    st.session_state["docs_panel"] = None  # let the panel decide again for this subject
    state.reset_chat()


def _select_chat(chat: ChatSummary, subject: str) -> None:
    try:
        messages = api_client.get_chat_messages(chat.chat_id)
    except ApiError as exc:
        st.session_state["flash"] = exc.message
        return
    state.new_view()
    st.session_state["chat_id"] = chat.chat_id
    st.session_state["chat_doc_id"] = chat.doc_id
    st.session_state["chat_messages"] = messages
    st.session_state["chat_subject"] = subject


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


def _open_new_subject() -> None:
    st.session_state["show_new_subject"] = True


def prompt_delete_subject() -> None:
    st.session_state["show_delete_subject"] = True


# -------------------------------------------------------------------- dialogs


@st.dialog("New collection")
def new_subject_dialog(existing_names: list[str]) -> None:
    # Clear the flag first so the dialog doesn't reopen on the next full rerun.
    st.session_state["show_new_subject"] = False
    name = st.text_input("Name", max_chars=60, placeholder="e.g. Biology 101, Q4 roadmap, onboarding docs")
    description = st.text_input("Description (optional)", max_chars=200, placeholder="What is this collection for?")
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
        st.session_state["docs_panel"] = True  # the next step is uploading notes
        state.reset_chat()
        st.rerun()


@st.dialog("Delete collection")
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
        st.markdown('<div class="brand">◈ Manifold</div>', unsafe_allow_html=True)
        st.button(
            "New chat",
            icon=":material/edit_square:",
            key="new_chat",
            use_container_width=True,
            on_click=state.reset_chat,
        )

        # ---- subjects
        st.markdown('<div class="side-label">Collections</div>', unsafe_allow_html=True)
        for index, subject in enumerate(subjects):
            selected = subject.name == current
            count = ready_counts.get(subject.name, 0)
            label = f"{subject.name}  ·  {count}" if count else subject.name
            row, menu = st.columns([6, 1], vertical_alignment="center")
            row.button(
                label,
                key=f"subject_btn_{index}",
                icon=":material/folder_open:" if selected else ":material/folder:",
                type="secondary" if selected else "tertiary",
                use_container_width=True,
                on_click=_select_subject,
                args=(subject.name,),
            )
            if selected:
                with menu.popover("", icon=":material/more_horiz:", type="tertiary"):
                    st.button(
                        "Delete collection",
                        icon=":material/delete:",
                        key="delete_subject_btn",
                        on_click=prompt_delete_subject,
                    )
        st.button(
            "New collection",
            icon=":material/add:",
            key="new_subject_btn",
            type="tertiary",
            use_container_width=True,
            on_click=_open_new_subject,
        )

        # ---- chats in the selected subject
        if current:
            st.markdown('<div class="side-label">Chats</div>', unsafe_allow_html=True)
            try:
                chats = api_client.list_chats(current)
            except ApiError as exc:
                chats = []
                st.caption(exc.message)
            chats = sorted(chats, key=lambda c: str(c.updated_at or ""), reverse=True)
            if not chats:
                st.caption("No chats yet in this collection.")
            for chat in chats[:30]:
                selected = chat.chat_id == st.session_state.get("chat_id")
                title = chat.title if len(chat.title) <= 28 else chat.title[:27] + "…"
                row, menu = st.columns([6, 1], vertical_alignment="center")
                row.button(
                    title,
                    key=f"chat_btn_{chat.chat_id}",
                    type="secondary" if selected else "tertiary",
                    use_container_width=True,
                    on_click=_select_chat,
                    args=(chat, current),
                )
                with menu.popover("", icon=":material/more_horiz:", type="tertiary"):
                    st.button(
                        "Delete chat",
                        icon=":material/delete:",
                        key=f"chat_del_{chat.chat_id}",
                        on_click=_delete_chat,
                        args=(chat.chat_id,),
                    )

        # ---- account (pinned to the bottom of the sidebar)
        with st.container(key="sidebar_user"):
            with st.popover(
                user.get("email") or "Account",
                icon=":material/account_circle:",
                use_container_width=True,
            ):
                st.caption("Theme: open the ⋮ menu (top right), then Settings.")
                st.button("Log out", icon=":material/logout:", on_click=_logout, use_container_width=True)

    if st.session_state.get("show_new_subject"):
        new_subject_dialog(names)
