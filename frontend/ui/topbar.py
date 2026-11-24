from __future__ import annotations

import streamlit as st

from core.api_client import DocumentRecord


def _toggle_docs(open_now: bool) -> None:
    st.session_state["docs_panel"] = not open_now


def render_topbar(subject: str, subject_docs: list[DocumentRecord], docs_open: bool) -> None:
    doc_names = {d.doc_id: d.filename for d in subject_docs}
    chat_id = st.session_state.get("chat_id")
    scope = None
    if chat_id:
        scope_doc = st.session_state.get("chat_doc_id")
        if scope_doc:
            scope = doc_names.get(scope_doc, scope_doc)

    with st.container(key="topbar"):
        left, right = st.columns([7, 2.4], vertical_alignment="center")
        with left:
            st.markdown(f"#### {subject}")
            if scope:
                st.caption(f"Focused on: {scope}")
        with right:
            st.button(
                f"Documents ({len(subject_docs)})",
                icon=":material/description:",
                key="docs_toggle",
                on_click=_toggle_docs,
                args=(docs_open,),
                type="secondary",
                use_container_width=True,
            )