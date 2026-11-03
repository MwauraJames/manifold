from __future__ import annotations

import streamlit as st

from core import api_client
from core.api_client import ApiError, DocumentRecord
from core.utils import ACTIVE_STATUSES
from ui import state
from ui.state import ALL_DOCS


def render_chat(subject_name: str, subject_docs: list[DocumentRecord]) -> None:
    # A different subject means a different conversation.
    if st.session_state.get("chat_subject") != subject_name:
        state.reset_chat()
        st.session_state["chat_subject"] = subject_name

    ready_docs = [d for d in subject_docs if d.status == "ready"]
    processing = any(d.status in ACTIVE_STATUSES for d in subject_docs)
    doc_names = {d.doc_id: d.filename for d in subject_docs}
    chat_id = st.session_state.get("chat_id")

    # ---- scope: which documents the question is about
    if chat_id:
        # The backend stores a chat's scope when it's created, so it's read-only now.
        scope_doc = st.session_state.get("chat_doc_id")
        label = doc_names.get(scope_doc, scope_doc) if scope_doc else "All documents in this subject"
        st.caption(f"Scope: {label}")
        if scope_doc and scope_doc not in doc_names:
            st.warning("The document this chat was about has been deleted.")
        doc_id = scope_doc
    else:
        options = [ALL_DOCS] + [d.doc_id for d in ready_docs]
        if st.session_state.get("scope_doc") not in options:
            st.session_state["scope_doc"] = ALL_DOCS
        choice = st.selectbox(
            "Ask about",
            options,
            format_func=lambda v: "All documents in this subject" if v == ALL_DOCS else doc_names.get(v, v),
            key="scope_doc",
        )
        doc_id = None if choice == ALL_DOCS else choice

    # ---- history
    messages = st.session_state["chat_messages"]
    if not messages:
        st.markdown("#### What would you like to know?")
        st.caption("Ask a question about your notes, ask for a summary, or ask to be quizzed.")
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---- input
    if not ready_docs:
        if processing:
            st.info("Your documents are still queued or being processed. You can ask questions once one is ready.")
        else:
            st.info("Upload a PDF in the Documents view to start asking questions.")
    prompt = st.chat_input("Ask a question about your notes…", disabled=not ready_docs)
    if prompt:
        _answer(prompt, subject_name, doc_id)


def _answer(prompt: str, subject: str, doc_id: str | None) -> None:
    messages = st.session_state["chat_messages"]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    meta: dict = {}
    errors: list[str] = []
    collected: list[str] = []

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("_Searching your documents…_")

        def tokens():
            first = True
            for event in api_client.ask_stream(
                prompt,
                subject,
                doc_id=doc_id,
                chat_id=st.session_state.get("chat_id"),
                model=st.session_state.get("model_id"),
            ):
                if event.kind == "meta":
                    meta.update(event.data)
                elif event.kind == "token":
                    if first:
                        status.empty()
                        first = False
                    content = event.data.get("content") or ""
                    collected.append(content)
                    yield content
                elif event.kind == "error":
                    errors.append(event.data.get("detail") or "Generation failed.")

        failure: ApiError | None = None
        try:
            st.write_stream(tokens())
        except ApiError as exc:
            failure = exc
        status.empty()

        answer = "".join(collected)
        if failure and not meta:
            # The request never started, so the backend saved nothing.
            messages.pop()
            st.error(failure.message)
            return
        if failure or errors:
            st.warning("The answer was interrupted. You can ask again.")

    if answer.strip():
        messages.append({"role": "assistant", "content": answer})
    if meta.get("chat_id") and not st.session_state.get("chat_id"):
        st.session_state["chat_id"] = meta["chat_id"]
        st.session_state["chat_doc_id"] = meta.get("doc_id") or doc_id
        st.session_state["chat_subject"] = subject
    # Rerun so the sidebar shows the new chat title.
    st.rerun()
