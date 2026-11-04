from __future__ import annotations

import html

import streamlit as st

from core import api_client
from core.api_client import ApiError, DocumentRecord
from core.stream_job import JobSnapshot, StreamJob
from core.utils import ACTIVE_STATUSES
from ui import state
from ui.state import ALL_DOCS

POLL_S = 0.25  # how often the answer area refreshes while text is streaming in
PLACEHOLDER = "Ask Manifold…"
SUGGESTIONS = [
    "Summarise the key points",
    "Quiz me on this material",
    "Explain the hardest concept simply",
]


# ------------------------------------------------------------------ helpers


def _model_label(model_id: str | None) -> str | None:
    if not model_id:
        return None
    models = st.session_state.get("_models")
    if models:
        for option in models.models:
            if model_id in (option.id, option.name):
                return option.name
    return model_id


def _matches(job: StreamJob, subject: str) -> bool:
    """Is this job's answer meant for the chat currently on screen?"""
    return job.view_id == st.session_state.get("chat_view_id") and job.subject == subject


def _commit(job: StreamJob, snap: JobSnapshot) -> None:
    """A finished answer becomes part of the on-screen history."""
    messages = st.session_state["chat_messages"]
    if job.kind == "edit" and len(messages) >= 2:
        del messages[-2:]  # the backend deleted the old question and answer
    messages.append({"role": "user", "content": job.prompt})
    messages.append({"role": "assistant", "content": snap.text, "model": snap.meta.get("model")})
    chat_id = snap.meta.get("chat_id")
    if chat_id and not st.session_state.get("chat_id"):
        st.session_state["chat_id"] = chat_id
        st.session_state["chat_doc_id"] = snap.meta.get("doc_id") or job.doc_id
        st.session_state["chat_subject"] = job.subject


def _reconcile_job(subject: str) -> StreamJob | None:
    """
    Called at the top of every render. Finished answers are committed to the
    history; jobs that belong to a chat the user has left are dropped (the
    backend has saved the answer, so it shows up when that chat is reopened).
    """
    job = st.session_state.get("stream_job")
    if job is None:
        return None
    snap = job.snapshot()
    here = _matches(job, subject)
    if snap.state == "done":
        if here:
            _commit(job, snap)
        st.session_state["stream_job"] = None
        return None
    if snap.state == "failed" and not here:
        st.session_state["stream_job"] = None
        return None
    return job


def _start_job(kind: str, prompt: str, subject: str, doc_id: str | None) -> bool:
    try:
        # Refresh an expired token here, in the main thread, before the
        # background thread starts using it.
        api_client.ensure_session()
    except ApiError as exc:
        if exc.status_code == 401:
            state.reset_all()
            st.session_state["auth_notice"] = "Your session expired. Please log in again."
            st.rerun()
        st.error(exc.message)
        return False

    job = StreamJob(
        kind=kind,
        prompt=prompt,
        subject=subject,
        doc_id=doc_id,
        chat_id=st.session_state.get("chat_id"),
        model=st.session_state.get("model_id"),
        view_id=st.session_state["chat_view_id"],
    )
    job.start(api_client.current_access_token())
    st.session_state["stream_job"] = job
    return True


def _queue_prompt(text: str) -> None:
    st.session_state["pending_prompt"] = text


# ---------------------------------------------------------------- main view


def render_chat(subject_name: str, subject_docs: list[DocumentRecord]) -> None:
    # A different subject means a different conversation.
    if st.session_state.get("chat_subject") != subject_name:
        state.reset_chat()
        st.session_state["chat_subject"] = subject_name

    job = _reconcile_job(subject_name)
    here = job is not None and _matches(job, subject_name)
    snap = job.snapshot() if job else None
    running_here = bool(here and snap.state == "running")
    running_elsewhere = job is not None and not here and snap.state == "running"
    failed_here = bool(here and snap.state == "failed")

    ready_docs = [d for d in subject_docs if d.status == "ready"]
    processing = any(d.status in ACTIVE_STATUSES for d in subject_docs)
    doc_names = {d.doc_id: d.filename for d in subject_docs}
    chat_id = st.session_state.get("chat_id")
    input_disabled = (not ready_docs) or running_here or running_elsewhere

    messages = st.session_state["chat_messages"]
    # While an edit is being regenerated (or failed after the backend already
    # deleted the old pair), hide the old question and answer.
    hide_tail = bool(
        job is not None
        and here
        and job.kind == "edit"
        and not (failed_here and snap.error is not None and snap.error.status_code == 400)
    )
    visible = messages[:-2] if hide_tail and len(messages) >= 2 else messages
    can_act = (
        bool(chat_id)
        and job is None
        and len(messages) >= 2
        and messages[-1]["role"] == "assistant"
        and messages[-2]["role"] == "user"
    )

    # ---- what to show
    new_chat_screen = not visible and not here
    typed = None
    if new_chat_screen:
        typed, doc_id = _render_hero(subject_name, ready_docs, doc_names, input_disabled, bool(chat_id))
    else:
        doc_id = st.session_state.get("chat_doc_id") if chat_id else (job.doc_id if job else None)
        _render_history(visible, can_act, subject_name, doc_names)
        if here:
            if running_here:
                # Poll the background job. Nothing in this fragment can interrupt it.
                runner = st.fragment(run_every=POLL_S)(_job_fragment)
                runner(job)
            elif failed_here:
                _render_failure(job, snap)
        # While an answer is being written the stop bar takes the composer's place.
        if not running_here:
            typed = st.chat_input(PLACEHOLDER, key="chat_prompt", disabled=input_disabled)

    # ---- notices
    if running_elsewhere:
        st.info("An answer is still being written in another chat. Wait for it to finish.")
    if not ready_docs:
        if processing:
            st.info("Your documents are still queued or being processed. You can ask once one is ready.")
        else:
            st.info("Add a document in the Documents panel to start asking questions.")

    prompt = st.session_state.pop("pending_prompt", None) or typed
    if prompt and not input_disabled and _start_job("ask", prompt, subject_name, doc_id):
        st.rerun()


# --------------------------------------------------------- new chat screen


def _render_hero(
    subject: str,
    ready_docs: list[DocumentRecord],
    doc_names: dict[str, str],
    disabled: bool,
    existing_chat: bool,
) -> tuple[str | None, str | None]:
    """Centered greeting, input and suggestions, like a fresh Gemini chat."""
    with st.container(key="hero"):
        st.markdown(
            f'<div class="hero-title">Where should we start?</div>'
            f'<div class="hero-sub">Ask anything about <b>{html.escape(subject)}</b></div>',
            unsafe_allow_html=True,
        )
        # Inside a container, st.chat_input stays in place instead of pinning to the bottom.
        typed = st.chat_input(PLACEHOLDER, key="chat_prompt", disabled=disabled)

        if existing_chat:
            scope_doc = st.session_state.get("chat_doc_id")
            label = doc_names.get(scope_doc, scope_doc) if scope_doc else "All documents"
            st.caption(f"About: {label}")
            doc_id = scope_doc
        else:
            options = [ALL_DOCS] + [d.doc_id for d in ready_docs]
            if st.session_state.get("scope_doc") not in options:
                st.session_state["scope_doc"] = ALL_DOCS
            _, middle, _ = st.columns([1, 3, 1])
            with middle:
                choice = st.selectbox(
                    "Ask about",
                    options,
                    format_func=lambda v: "Ask about: all documents" if v == ALL_DOCS else f"Ask about: {doc_names.get(v, v)}",
                    key="scope_doc",
                    label_visibility="collapsed",
                )
            doc_id = None if choice == ALL_DOCS else choice

        if not disabled:
            with st.container(key="suggestions"):
                for index, text in enumerate(SUGGESTIONS):
                    st.button(
                        text,
                        key=f"suggest_{index}",
                        icon=":material/subdirectory_arrow_right:",
                        type="tertiary",
                        on_click=_queue_prompt,
                        args=(text,),
                    )
    return typed, doc_id


# ------------------------------------------------------------ conversation


def _user_bubble(text: str, key: str) -> None:
    with st.container(key=key):
        st.markdown(text)


def _copy_popover(text: str, key: str) -> None:
    # st.code has a built-in copy button.
    with st.popover("", icon=":material/content_copy:", type="tertiary", help="Copy", key=key):
        st.code(text, language=None, wrap_lines=True)


def _render_history(
    visible: list[dict],
    can_act: bool,
    subject: str,
    doc_names: dict[str, str],
) -> None:
    for index, message in enumerate(visible):
        is_last_question = can_act and index == len(visible) - 2
        is_last_answer = can_act and index == len(visible) - 1

        if message["role"] == "user":
            if is_last_question and st.session_state.get("editing_last"):
                _render_edit_form(subject)
                continue
            _user_bubble(message["content"], f"user_msg_{index}")
            if is_last_question:
                with st.container(key="user_actions", horizontal=True, horizontal_alignment="right"):
                    st.button(
                        "",
                        icon=":material/edit:",
                        key="edit_last_btn",
                        type="tertiary",
                        help="Edit prompt",
                        on_click=_begin_edit,
                        args=(message["content"],),
                    )
                    _copy_popover(message["content"], "copy_user")
        else:
            with st.container(key=f"asst_msg_{index}"):
                st.markdown(message["content"])
                if message.get("model"):
                    st.caption(f"Answered by {_model_label(message['model'])}")
            if is_last_answer:
                question = visible[index - 1]["content"]
                with st.container(key="asst_actions", horizontal=True):
                    _copy_popover(message["content"], "copy_asst")
                    if st.button(
                        "",
                        icon=":material/refresh:",
                        key="regen_btn",
                        type="tertiary",
                        help="Regenerate response",
                    ):
                        # The backend replaces the last answer when the same question is "edited".
                        if _start_job("edit", question, subject, st.session_state.get("chat_doc_id")):
                            st.rerun()


# ------------------------------------------------------------ edit last message


def _begin_edit(text: str) -> None:
    st.session_state["edit_text"] = text
    st.session_state["editing_last"] = True


def _cancel_edit() -> None:
    st.session_state["editing_last"] = False


def _render_edit_form(subject: str) -> None:
    st.text_area("Edit your message", key="edit_text", height=120, label_visibility="collapsed")
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("Cancel", on_click=_cancel_edit)
        if st.button("Update", type="primary"):
            text = (st.session_state.get("edit_text") or "").strip()
            if not text:
                st.error("The message can't be empty.")
                return
            st.session_state["editing_last"] = False
            if _start_job("edit", text, subject, st.session_state.get("chat_doc_id")):
                st.rerun()


# --------------------------------------------------------- the live answer


def _job_fragment(job: StreamJob) -> None:
    snap = job.snapshot()
    if snap.state != "running":
        # Hand over to the main script, which commits the answer or shows the error.
        st.rerun(scope="app")

    _user_bubble(job.prompt, "user_msg_live")
    with st.container(key="asst_msg_live"):
        if snap.text:
            st.markdown(snap.text + " ▌")

    # The stop bar replaces the composer while an answer is written. Stop only works
    # once the model is writing: the backend resets its stop flag when generation
    # begins, so a click during the search phase would be lost.
    chat_id = snap.meta.get("chat_id") or job.chat_id
    can_stop = bool(snap.text and chat_id)
    with st.container(key="stopbar", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown("Writing…" if snap.text else "Searching your documents…")
        clicked = st.button(
            "Stopping…" if job.stop_requested else "Stop",
            key="stop_generation",
            icon=":material/stop_circle:",
            type="primary",
            disabled=job.stop_requested or not can_stop,
            help="Stop generating",
        )
    if clicked and can_stop:
        job.stop_requested = True
        try:
            api_client.stop_generation(chat_id)
        except ApiError as exc:
            job.stop_requested = False
            st.toast(exc.message)


def _render_failure(job: StreamJob, snap: JobSnapshot) -> None:
    error = snap.error
    _user_bubble(job.prompt, "user_msg_failed")
    with st.container(key="asst_msg_failed"):
        if snap.text:
            st.markdown(snap.text)
        if error is not None and error.status_code == 400:
            # Most likely the chosen model is no longer available: refresh the list.
            st.session_state.pop("_models", None)
            st.error(error.message)
        elif snap.meta or snap.text:
            st.warning("The answer was interrupted, so it wasn't saved. Send the message again to retry.")
        else:
            st.error(error.message if error else "Something went wrong.")
        if job.kind == "edit" and not (error is not None and error.status_code == 400):
            st.caption("Your earlier answer to this message was removed when the edit started.")
