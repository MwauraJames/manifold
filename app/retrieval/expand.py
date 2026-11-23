# app/retrieval/expand.py
import json
import os
import re
from typing import Literal
from langsmith import traceable
from generation.prompts import QUERY_EXPANSION_PROMPT
from generation.models import invoke_llm_traced
from configs.config import Settings

settings = Settings()
EXPAND_MODEL = getattr(settings, "expand_model", None)

QueryIntent = Literal["fact_search", "global_synthesis"]


def _get_fast_expand_model(explicit_model: str | None = None) -> str | None:
    """
    Picks a fast, low-latency model for routing and query rewriting so we never
    accidentally default to a heavy reasoning model (e.g., gpt-oss-120b).
    """
    if explicit_model:
        return explicit_model
    if EXPAND_MODEL:
        return EXPAND_MODEL

    env_expand = os.getenv("EXPAND_MODEL", "").strip()
    if env_expand:
        return env_expand

    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq/openai/gpt-oss-20b"
    if os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip():
        return "gemini/gemini-3.8-flash"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai/gpt-4.1-mini"

    return None


def _parse_router_response(raw_output: str, fallback_query: str) -> dict[str, str]:
    """Extracts JSON from the LLM output safely and treats the LLM's intent as final."""
    cleaned = raw_output.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            intent = str(data.get("intent", "")).strip()
            if intent not in ("fact_search", "global_synthesis"):
                intent = "fact_search"

            expanded = (data.get("expanded_query") or fallback_query).strip()
            return {"intent": intent, "expanded_query": expanded}
        except json.JSONDecodeError:
            pass

    plain_expanded = cleaned if cleaned and not cleaned.startswith("{") else fallback_query
    return {
        "intent": "fact_search",
        "expanded_query": plain_expanded,
    }


@traceable(name="route_and_expand_query", run_type="chain")
def route_and_expand_query(
    query: str,
    chat_history: list[dict] | None = None,
    model_name: str | None = None,
    summary: str | None = None,
) -> dict[str, str]:
    """
    Sends every query (including Turn 1) to the router/expansion LLM to:
      1. Decide the final retrieval intent ('fact_search' vs 'global_synthesis').
      2. Rewrite/expand the user's question into a self-contained search query.
    Keeps `summary` separate from `chat_history[-6:]` so the summary is never sliced off.
    """
    clean_query = query.strip()

    raw_turns = [
        m
        for m in (chat_history or [])
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
    recent_slice = raw_turns[-6:]

    prompt_parts: list[str] = []
    if summary and summary.strip():
        prompt_parts.append(f"Summary of Earlier Conversation:\n{summary.strip()}")

    if recent_slice:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent_slice
        )
        prompt_parts.append(f"Recent Conversation History:\n{history_text}")

    if not prompt_parts:
        prompt_parts.append("Conversation History:\n(None — first turn)")

    prompt_parts.append(f"User's Latest Question: {clean_query}")
    user_prompt = "\n\n".join(prompt_parts)

    messages = [
        {"role": "system", "content": QUERY_EXPANSION_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    target_model = _get_fast_expand_model(model_name)

    try:
        raw_output = invoke_llm_traced(
            model_name=target_model,
            messages=messages,
            timeout=15,
        ).strip()
        return _parse_router_response(raw_output, fallback_query=clean_query)
    except Exception as e:
        print(f"[route_and_expand_query] Fallback due to error: {e}")
        return {
            "intent": "fact_search",
            "expanded_query": clean_query,
        }


def expand_query(
    query: str,
    chat_history: list[dict] | None = None,
    model_name: str | None = None,
    summary: str | None = None,
) -> str:
    """Backward-compatible helper returning only the expanded query string."""
    return route_and_expand_query(
        query,
        chat_history=chat_history,
        model_name=model_name,
        summary=summary,
    )["expanded_query"]