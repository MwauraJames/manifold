# frontend/ui/chat_ui.py
from __future__ import annotations

import html
import re

import streamlit as st

from core import api_client
from core.api_client import ApiError, DocumentRecord
from core.stream_job import JobSnapshot, StreamJob
from core.utils import ACTIVE_STATUSES
from ui import state
from ui.state import ALL_DOCS

POLL_S = 0.25
PLACEHOLDER = "Ask Manifold…"
SUGGESTIONS = [
    "Summarise the key points",
    "Quiz me on this material",
    "List the action items and deadlines",
]

_PROTECTED_SPAN_RE = re.compile(
    r"(```[\s\S]*?```|\$\$[\s\S]*?\$\$|`[^`\n]+`|\$[^$\n]+\$)"
)
_CITE_BRACKET_RE = re.compile(r"\[(\d{1,2})\]")


def _model_label(model_id: str | None) -> str | None:
    if not model_id:
        return None
    models = st.session_state.get("_models")
    if models:
        for option in models.models:
            if model_id in (option.id, option.name):
                return option.name
    return model_id


def _extract_citations(snap: JobSnapshot) -> list[dict]:
    cites = getattr(snap, "citations", None)
    if cites is None and isinstance(snap.meta, dict):
        cites = snap.meta.get("citations")
    return list(cites) if cites else []


def _decorate_inline_citations(
    text: str,
    citations: list[dict] | None,
    doc_names: dict[str, str] | None = None,
) -> str:
    if not text or not citations:
        return text

    cite_map: dict[int, dict] = {}
    for c in citations:
        try:
            cid = int(c.get("id"))
            cite_map[cid] = c
        except (TypeError, ValueError):
            continue

    if not cite_map:
        return text

    def _replace_marker(match: re.Match) -> str:
        num = int(match.group(1))
        c = cite_map.get(num)
        if not c:
            return match.group(0)
        doc_id = c.get("doc_id")
        fname = (
            c.get("filename")
            or (doc_names.get(doc_id) if doc_names and doc_id else None)
            or doc_id
            or f"Source {num}"
        )
        section = c.get("section") or ""
        score = c.get("score")
        tooltip_parts = [str(fname)]
        if section:
            tooltip_parts.append(str(section))
        if score is not None:
            tooltip_parts.append(f"Score: {score:.2f}")
        tooltip = html.escape(" • ".join(tooltip_parts), quote=True)
        return f'<span class="cite-badge" title="{tooltip}">[{num}]</span>'

    parts = _PROTECTED_SPAN_RE.split(text)
    for i in range(0, len(parts), 2):
        parts[i] = _CITE_BRACKET_RE.sub(_replace_marker, parts[i])
    return "".join(parts)


def _render_citations_drawer(
    citations: list[dict] | None,
    doc_names: dict[str, str] | None = None,
) -> None:
    if not citations:
        return

    cited_list = [c for c in citations if c.get("cited")]
    ordered = cited_list + [c for c in citations if not c.get("cited")]

    if cited_list:
        label = (
            f"Sources ({len(cited_list)} cited"
            + (f" • {len(citations)} retrieved)" if len(citations) > len(cited_list) else ")")
        )
    else:
        label = f"Retrieved Context ({len(citations)} passages)"

    with st.expander(label, expanded=False, icon=":material/menu_book:"):
        for c in ordered:
            cid = c.get("id", "?")
            doc_id = c.get("doc_id")
            fname = (
                c.get("filename")
                or (doc_names.get(doc_id) if doc_names and doc_id else None)
                or doc_id
                or f"Source {cid}"
            )
            section = c.get("section") or "Passage"
            score = c.get("score")
            mode = c.get("retrieval_mode") or "fact_search"
            is_cited = bool(c.get("cited"))
            snippet = (c.get("snippet") or "").strip()

            card_cls = "source-card is-cited" if is_cited else "source-card"
            badge_html = (
                '<span class="source-pill cited-pill">Cited in answer</span>'
                if is_cited
                else '<span class="source-pill">Retrieved</span>'
            )
            score_html = (
                f'<span class="source-pill">Relevance: {score:.2f}</span>'
                if isinstance(score, (int, float))
                else (
                    '<span class="source-pill">Synthesis Sample</span>'
                    if mode == "global_synthesis"
                    else ""
                )
            )

            st.markdown(
                f'<div class="{card_cls}">'
                f'  <div class="source-card-header">'
                f'    <div class="source-card-title">'
                f'      <span class="cite-badge">[{html.escape(str(cid))}]</span> '
                f'      {html.escape(str(fname))} '
                f'      <span style="font-weight:400; opacity:0.75;">— {html.escape(str(section))}</span>'
                f'    </div>'
                f'    <div class="source-card-meta">{badge_html}{score_html}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if snippet:
                st.caption(snippet)


def _matches(job: StreamJob, subject: str) -> bool:
    return job.view_id == st.session_state.get("chat_view_id") and job.subject == subject


def _commit(job: StreamJob, snap: JobSnapshot) -> None:
    messages = st.session_state["chat_messages"]
    if job.kind == "edit" and len(messages) >= 2:
        del messages[-2:]
    messages.append({"role": "user", "content": job.prompt})
    messages.append(
        {
            "role": "assistant",
            "content": snap.text,
            "model": snap.meta.get("model"),
            "citations": _extract_citations(snap),
        }
    )
    chat_id = snap.meta.get("chat_id")
    if chat_id and not st.session_state.get("chat_id"):
        st.session_state["chat_id"] = chat_id
        st.session_state["chat_subject"] = job.subject

    committed_doc_id = snap.meta.get("doc_id") if "doc_id" in snap.meta else job.doc_id
    st.session_state["chat_doc_id"] = committed_doc_id


def _reconcile_job(subject: str) -> StreamJob | None:
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


def _on_scope_change() -> None:
    """Keeps chat_doc_id synced before render_topbar() runs on the next rerun."""
    choice = st.session_state.get("scope_doc", ALL_DOCS)
    st.session_state["chat_doc_id"] = None if choice == ALL_DOCS else choice


def _render_scope_selector(
    ready_docs: list[DocumentRecord],
    doc_names: dict[str, str],
    disabled: bool,
) -> str | None:
    options = [ALL_DOCS] + [d.doc_id for d in ready_docs]
    desired = st.session_state.get("chat_doc_id") or ALL_DOCS
    if desired not in options:
        desired = ALL_DOCS
        st.session_state["chat_doc_id"] = None
    if st.session_state.get("scope_doc") != desired:
        st.session_state["scope_doc"] = desired

    choice = st.selectbox(
        "Ask about",
        options,
        format_func=lambda v: (
            "Ask about: all documents"
            if v == ALL_DOCS
            else f"Ask about: {doc_names.get(v, v)}"
        ),
        key="scope_doc",
        disabled=disabled,
        label_visibility="collapsed",
        on_change=_on_scope_change,
    )
    return None if choice == ALL_DOCS else choice


def _render_chat_scope_bar(
    ready_docs: list[DocumentRecord],
    doc_names: dict[str, str],
    disabled: bool,
) -> str | None:
    if not ready_docs:
        return None
    with st.container(key="chat_scope_bar"):
        col, _ = st.columns([3, 2])
        with col:
            return _render_scope_selector(ready_docs, doc_names, disabled)


def render_chat(subject_name: str, subject_docs: list[DocumentRecord]) -> None:
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
    active_docs = [d for d in subject_docs if d.status in ACTIVE_STATUSES]
    doc_names = {d.doc_id: d.filename for d in subject_docs}
    chat_id = st.session_state.get("chat_id")
    input_disabled = (not ready_docs) or running_here or running_elsewhere

    messages = st.session_state["chat_messages"]
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

    new_chat_screen = not visible and not here
    typed = None
    if new_chat_screen:
        typed, doc_id = _render_hero(subject_name, ready_docs, doc_names, input_disabled)
    else:
        _render_history(visible, can_act, subject_name, doc_names)
        if here:
            if running_here:
                runner = st.fragment(run_every=POLL_S)(_job_fragment)
                runner(job, doc_names)
            elif failed_here:
                _render_failure(job, snap)

        if not running_here:
            doc_id = _render_chat_scope_bar(ready_docs, doc_names, input_disabled)
            typed = st.chat_input(PLACEHOLDER, key="chat_prompt", disabled=input_disabled)
        else:
            doc_id = st.session_state.get("chat_doc_id")

    if running_elsewhere:
        st.info("An answer is still being written in another chat. Wait for it to finish.")

    if active_docs:
        names = ", ".join(f"**{d.filename}**" for d in active_docs[:3])
        if len(active_docs) > 3:
            names += f" and {len(active_docs) - 3} more"
        st.info(
            f"⏳ Processing {names}. Deep OCR, vision transcription, and hybrid indexing "
            "can take 1–2 minutes per file—feel free to wait or query already-ready documents.",
            icon=":material/hourglass_top:",
        )
    elif not ready_docs:
        st.info("Add a document in the Documents panel to start asking questions.")

    prompt = st.session_state.pop("pending_prompt", None) or typed
    if prompt and not input_disabled and _start_job("ask", prompt, subject_name, doc_id):
        st.rerun()


def _render_hero(
    subject: str,
    ready_docs: list[DocumentRecord],
    doc_names: dict[str, str],
    disabled: bool,
) -> tuple[str | None, str | None]:
    with st.container(key="hero"):
        st.markdown(
            f'<div class="hero-title">Where should we start?</div>'
            f'<div class="hero-sub">Ask anything about <b>{html.escape(subject)}</b></div>',
            unsafe_allow_html=True,
        )
        typed = st.chat_input(PLACEHOLDER, key="chat_prompt", disabled=disabled)

        _, middle, _ = st.columns([1, 3, 1])
        with middle:
            doc_id = _render_scope_selector(ready_docs, doc_names, disabled)

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


def _user_bubble(text: str, key: str) -> None:
    with st.container(key=key):
        st.markdown(text)


def _copy_popover(text: str, key: str) -> None:
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
            citations = message.get("citations") or []
            rendered_md = _decorate_inline_citations(message["content"], citations, doc_names)
            with st.container(key=f"asst_msg_{index}"):
                st.markdown(rendered_md, unsafe_allow_html=True)
                if citations:
                    _render_citations_drawer(citations, doc_names)
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
                        if _start_job("edit", question, subject, st.session_state.get("chat_doc_id")):
                            st.rerun()


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


def _job_fragment(job: StreamJob, doc_names: dict[str, str] | None = None) -> None:
    snap = job.snapshot()
    if snap.state != "running":
        st.rerun(scope="app")

    live_citations = _extract_citations(snap)
    _user_bubble(job.prompt, "user_msg_live")
    with st.container(key="asst_msg_live"):
        if snap.text:
            live_md = _decorate_inline_citations(snap.text + " ▌", live_citations, doc_names)
            st.markdown(live_md, unsafe_allow_html=True)

    chat_id = snap.meta.get("chat_id") or job.chat_id
    can_stop = bool(snap.text and chat_id)
    with st.container(key="stopbar", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        if snap.text:
            status_text = (
                f"Writing answer… ({len(live_citations)} sources cited)"
                if live_citations
                else "Writing answer…"
            )
        else:
            status_text = getattr(snap, "status", None) or "Analyzing query & searching documents…"
        st.markdown(status_text)
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
            st.session_state.pop("_models", None)
            st.error(error.message)
        elif snap.meta or snap.text:
            st.warning("The answer was interrupted, so it wasn't saved. Send the message again to retry.")
        else:
            st.error(error.message if error else "Something went wrong.")
        if job.kind == "edit" and not (error is not None and error.status_code == 400):
            st.caption("Your earlier answer to this message was removed when the edit started.")