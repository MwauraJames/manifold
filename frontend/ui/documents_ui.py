from __future__ import annotations

import streamlit as st

from core import api_client
from core.api_client import ApiError, DocumentRecord
from core.config import (
    MAX_FILES_PER_BATCH,
    MAX_UPLOAD_MB,
    POLL_SECONDS,
    STALE_PROCESSING_MIN,
    STALE_QUEUED_MIN,
)
from core.utils import ACTIVE_STATUSES, ALLOWED_EXTENSIONS, format_date, minutes_since, plan_upload

STATUS_LABEL = {
    "queued": ":gray[Queued]",
    "processing": ":orange[Processing…]",
    "ready": ":green[Ready]",
    "failed": ":red[Failed]",
}


def _close_panel() -> None:
    st.session_state["docs_panel"] = False


def render_documents_panel(subject_name: str, subject_docs: list[DocumentRecord]) -> None:
    """The right-hand panel: upload files and manage the subject's documents."""
    with st.container(key="docs_panel"):
        title_col, close_col = st.columns([5, 1], vertical_alignment="center")
        title_col.markdown("#### Documents")
        close_col.button(
            "",
            icon=":material/close:",
            key="close_docs",
            type="tertiary",
            help="Close",
            on_click=_close_panel,
        )
        st.caption(f"Files in **{subject_name}**. Questions are answered from these.")
        _render_upload(subject_name, subject_docs)
        st.divider()

        active = any(d.status in ACTIVE_STATUSES for d in subject_docs)
        # Only poll while something is queued or processing. The fragment re-fetches
        # the list every few seconds without rerunning the whole app.
        runner = st.fragment(run_every=POLL_SECONDS if active else None)(_documents_fragment)
        runner(subject_name, active)


def _render_upload(subject: str, existing_docs: list[DocumentRecord]) -> None:
    version = st.session_state.get("uploader_version", 0)
    uploaded = st.file_uploader(
        "Add documents",
        type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploader_{version}",
        help=(
            f"Up to {MAX_FILES_PER_BATCH} files at a time, {MAX_UPLOAD_MB} MB each. "
            "They are processed one after another."
        ),
    )
    if not uploaded:
        return

    plan = plan_upload(uploaded, existing_docs, MAX_UPLOAD_MB, MAX_FILES_PER_BATCH)
    if plan.error:
        st.error(plan.error)
        return

    if plan.skipped:
        lines = "\n".join(f"- **{name}**: {reason}" for name, reason in plan.skipped)
        st.warning(f"These files will be skipped:\n\n{lines}")

    if not plan.accepted:
        return

    confirmed = True
    if plan.replacing:
        names = ", ".join(f"**{n}**" for n in plan.replacing)
        st.warning(
            f"These already exist in this subject: {names}. "
            f"Uploading will replace {'it' if len(plan.replacing) == 1 else 'them'}."
        )
        confirmed = st.checkbox("Replace the existing documents", key=f"replace_{version}")

    count = len(plan.accepted)
    label = "Upload 1 file" if count == 1 else f"Upload {count} files"
    if st.button(label, type="primary", disabled=not confirmed, use_container_width=True):
        payload = [(f.name, f.getvalue()) for f in plan.accepted]
        with st.spinner(f"Uploading {count} file{'s' if count != 1 else ''}…"):
            try:
                result = api_client.upload_files(payload, subject)
            except ApiError as exc:
                st.error(exc.message)
                if exc.status_code and exc.status_code >= 500:
                    # The server handles files one by one, so some may already be queued.
                    st.caption("Some files may already have been queued. Check the list below.")
                return
        queued = len(result.documents) or count
        st.session_state["uploader_version"] = version + 1  # clears the uploader
        st.session_state["flash"] = (
            f"{queued} file{'s' if queued != 1 else ''} queued. They'll be processed one at a time."
        )
        st.rerun()


def _documents_fragment(subject: str, was_active: bool) -> None:
    try:
        docs = api_client.list_documents(subject)
    except ApiError as exc:
        st.error(exc.message)
        return

    queued_n = sum(1 for d in docs if d.status == "queued")
    processing_n = sum(1 for d in docs if d.status == "processing")
    if was_active and not (queued_n or processing_n):
        # Everything finished: rerun the whole app so the chat and the counts
        # see the new "ready" documents.
        st.rerun(scope="app")

    if not docs:
        st.caption("No documents yet. Add some above to start asking questions.")
        return

    if queued_n or processing_n:
        parts = []
        if processing_n:
            parts.append(f"{processing_n} processing")
        if queued_n:
            parts.append(f"{queued_n} queued")
        st.caption(" · ".join(parts).capitalize() + ". Updates automatically.")

    for index, doc in enumerate(docs):
        name_col, menu_col = st.columns([7, 1], vertical_alignment="center")
        with name_col:
            st.markdown(f"**{doc.filename}**")
            status_line = STATUS_LABEL.get(doc.status, doc.status)
            uploaded_at = format_date(doc.created_at)
            st.caption(f"{status_line} · {uploaded_at}" if uploaded_at else status_line)
            if doc.status == "failed" and doc.error:
                st.caption(doc.error[:140])
            age = minutes_since(doc.updated_at)
            if doc.status == "processing" and age is not None and age > STALE_PROCESSING_MIN:
                st.caption("Taking longer than expected. You can delete it and try again.")
            if doc.status == "queued" and age is not None and age > STALE_QUEUED_MIN:
                st.caption(
                    "Still waiting. If the server was restarted, the queue was lost: "
                    "delete this and upload it again."
                )
        with menu_col:
            with st.popover("", icon=":material/more_vert:", type="tertiary"):
                st.write("Delete this document?")
                if st.button("Delete", key=f"del_doc_{index}_{doc.doc_id}", type="primary"):
                    try:
                        api_client.delete_document(doc.doc_id)
                    except ApiError as exc:
                        if exc.status_code != 404:
                            st.error(exc.message)
                            continue
                    st.rerun(scope="app")
