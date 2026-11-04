from __future__ import annotations

import time

import streamlit as st

from core import api_client
from core.api_client import ApiError, ModelList

MODELS_TTL_S = 300  # re-check the server's model list every 5 minutes


def load_models() -> ModelList:
    """The server's model list, cached per browser session for a few minutes."""
    cached = st.session_state.get("_models")
    age = time.time() - st.session_state.get("_models_at", 0)
    if cached is not None and age < MODELS_TTL_S:
        return cached
    try:
        models = api_client.list_models()
    except ApiError:
        # Keep showing the last good list if a refresh fails.
        models = cached if cached is not None else ModelList()
    st.session_state["_models"] = models
    st.session_state["_models_at"] = time.time()
    return models


def render_model_picker() -> None:
    models = load_models()
    if not models.models:
        st.session_state["model_id"] = None
        return
    ids = [m.id for m in models.models]
    labels = {m.id: m.name for m in models.models}
    if st.session_state.get("model_id") not in ids:
        st.session_state["model_id"] = models.default_model if models.default_model in ids else ids[0]
    st.selectbox(
        "Model",
        ids,
        format_func=lambda i: labels[i],
        key="model_id",
        label_visibility="collapsed",
        help="Which model writes the answers",
    )
