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

# Explicit signals in the USER'S raw prompt that genuinely require global_synthesis
_GLOBAL_SYNTHESIS_SIGNALS_RE = re.compile(
    r"\b("
    r"exam|exams|quiz|quizzes|test\s+me|mock|flashcards?|"
    r"questions?\b.*?\b(unit|doc|document|lecture|pdf|course|notes|subject|chapter|exam|study|possible|practice)|"
    r"(possible|practice|sample|open[- ]ended|multiple[- ]choice|revision)\s+questions?|"
    r"generate\s+.*?\bquestions?|"
    r"ask\s+me\s+questions?|"
    r"(summarize|summary|overview|outline|key\s+takeaways|study\s+guide)\b.*?\b(unit|doc|document|lecture|pdf|course|notes|subject|everything|all|whole|entire)|"
    r"what\s+is\s+(this|the)\s+(unit|doc|document|lecture|pdf|course|notes|subject)\s+about|"
    r"explain\s+(this|the)\s+(unit|doc|document|lecture|pdf|course|notes|subject|everything|all|whole)"
    r")\b",
    re.IGNORECASE,
)

# Add near the top of app/retrieval/expand.py:
_NEEDS_CONTEXT_RE = re.compile(
    r"\b("
    r"it|its|they|them|their|this|that|these|those|he|she|"
    r"same|above|previous|prior|earlier|last|first|second|third|"
    r"more|another|else|again|further|elaborate|clarify|why|how\s+so|"
    r"what\s+about|and\s+for"
    r")\b",
    re.IGNORECASE,
)


def _needs_history_expansion(query: str) -> bool:
    """
    Returns True only if the query is very short (<4 words) or contains
    pronouns/referential words that depend on prior chat turns.
    """
    words = query.strip().split()
    if len(words) < 4:
        return True
    return bool(_NEEDS_CONTEXT_RE.search(query))


def _has_global_synthesis_signal(raw_query: str) -> bool:
    """
    Returns True ONLY if the user's raw query explicitly asks for study aids
    (exam/practice questions, quizzes, flashcards) or a whole-unit/document summary.
    Prevents bare topic queries like 'networking protocols' from being misrouted.
    """
    return bool(_GLOBAL_SYNTHESIS_SIGNALS_RE.search(raw_query.strip()))


def _get_fast_expand_model(explicit_model: str | None = None) -> str | None:
    """
    Picks an ultra-fast, low-latency model for query rewriting so we never
    accidentally default to the heavy 120B reasoning model (gpt-oss-120b).
    """
    if explicit_model:
        return explicit_model
    if EXPAND_MODEL:
        return EXPAND_MODEL

    env_expand = os.getenv("EXPAND_MODEL", "").strip()
    if env_expand:
        return env_expand

    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq/llama-3.1-8b-instant"
    if os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip():
        return "gemini/gemini-3.8-flash"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai/gpt-4.1-mini"

    return None


def _parse_router_response(raw_output: str, fallback_query: str) -> dict[str, str]:
    """Extracts JSON from the LLM output safely and enforces the topic guardrail."""
    cleaned = raw_output.strip()
    match = re.search(r"\{.*\}", cleaned, dotall := re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            intent = data.get("intent", "fact_search")
            if intent not in ("fact_search", "global_synthesis"):
                intent = (
                    "global_synthesis"
                    if _has_global_synthesis_signal(fallback_query)
                    else "fact_search"
                )

            if intent == "global_synthesis" and not _has_global_synthesis_signal(fallback_query):
                intent = "fact_search"

            expanded = (data.get("expanded_query") or fallback_query).strip()
            return {"intent": intent, "expanded_query": expanded}
        except json.JSONDecodeError:
            pass

    plain_expanded = cleaned if cleaned and not cleaned.startswith("{") else fallback_query
    return {
        "intent": (
            "global_synthesis"
            if _has_global_synthesis_signal(fallback_query)
            else "fact_search"
        ),
        "expanded_query": plain_expanded,
    }


@traceable(name="route_and_expand_query", run_type="chain")
def route_and_expand_query(
    query: str,
    chat_history: list[dict] | None = None,
    model_name: str | None = None,
) -> dict[str, str]:
    """
    Classifies the query intent ('fact_search' vs 'global_synthesis') and
    rewrites the user's question into a self-contained query.
    - Fast-Path 1 (<1ms): If the query matches global_synthesis signals, no Qdrant
      search is needed anyway—return immediately without an LLM call.
    - Fast-Path 2 (<1ms): If chat_history is empty (Turn 1), no coreference
      resolution is needed—return immediately without an LLM call.
    - Multi-Turn Path (~250ms): Uses llama-3.1-8b-instant to rewrite pronouns/context.
    """
    clean_query = query.strip()

    # Fast-Path 1: Explicit global synthesis request (skips Qdrant anyway)
    if _has_global_synthesis_signal(clean_query):
        return {"intent": "global_synthesis", "expanded_query": clean_query}

    # Fast-Path 2: No prior history OR the question is already 100% self-contained
    if not chat_history or not _needs_history_expansion(clean_query):
        return {"intent": "fact_search", "expanded_query": clean_query}

    # Fast-Path 2: First turn of conversation (no prior history to resolve)
    if not chat_history:
        return {"intent": "fact_search", "expanded_query": clean_query}

    # Multi-Turn Path: Only keep the last 6 messages for fast context rewriting
    recent_slice = chat_history[-6:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent_slice
    )
    user_prompt = (
        f"Conversation History:\n{history_text}\n\n"
        f"User's Latest Question: {clean_query}"
    )

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
) -> str:
    """Backward-compatible helper returning only the expanded query string."""
    return route_and_expand_query(query, chat_history, model_name)["expanded_query"]