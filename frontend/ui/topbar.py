from __future__ import annotations

import streamlit as st

from core.api_client import DocumentRecord
from ui import models


def _toggle_docs(open_now: bool) -> None:
    st.session_state["docs_panel"] = not open_now


def render_topbar(subject: str, subject_docs: list[DocumentRecord], docs_open: bool) -> None:
    doc_names = {d.doc_id: d.filename for d in subject_docs}
    chat_id = st.session_state.get("chat_id")
    scope = None
    if chat_id:
        scope_doc = st.session_state.get("chat_doc_id")
        scope = doc_names.get(scope_doc, scope_doc) if scope_doc else "All documents"

    with st.container(key="topbar"):
        left, middle, right = st.columns([5, 3, 3], vertical_alignment="center")
        _render_topbar_cells(subject, scope, docs_open, len(subject_docs), left, middle, right)


def _render_topbar_cells(subject, scope, docs_open, doc_count, left, middle, right) -> None:
    with left:
        st.markdown(f"#### {subject}")
        if scope:
            st.caption(f"About: {scope}")
    with middle:
        models.render_model_picker()
    with right:
        st.button(
            f"Documents ({doc_count})",
            icon=":material/description:",
            key="docs_toggle",
            on_click=_toggle_docs,
            args=(docs_open,),
            type="primary" if docs_open else "secondary",
            use_container_width=True,
        )
