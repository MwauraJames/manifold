# app/retrieval/expand.py
import json
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
    # New additions for natural language unit/document queries:
    r"what\s+is\s+(this|the)\s+(unit|doc|document|lecture|pdf|course|notes|subject)\s+about|"
    r"explain\s+(this|the)\s+(unit|doc|document|lecture|pdf|course|notes|subject|everything|all|whole)"
    r")\b",
    re.IGNORECASE,
)


def _has_global_synthesis_signal(raw_query: str) -> bool:
    """
    Returns True ONLY if the user's raw query explicitly asks for study aids
    (exam/practice questions, quizzes, flashcards) or a whole-unit/document summary.
    Prevents bare topic queries like 'networking protocols' from being misrouted.
    """
    return bool(_GLOBAL_SYNTHESIS_SIGNALS_RE.search(raw_query.strip()))


def _parse_router_response(raw_output: str, fallback_query: str) -> dict[str, str]:
    """Extracts JSON from the LLM output safely and enforces the topic guardrail."""
    cleaned = raw_output.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            intent = data.get("intent", "fact_search")
            if intent not in ("fact_search", "global_synthesis"):
                intent = "global_synthesis" if _has_global_synthesis_signal(fallback_query) else "fact_search"

            # Deterministic guardrail: if the LLM says 'global_synthesis' because it added
            # the word 'overview' to expanded_query, verify against the user's raw query.
            if intent == "global_synthesis" and not _has_global_synthesis_signal(fallback_query):
                intent = "fact_search"

            expanded = (data.get("expanded_query") or fallback_query).strip()
            return {"intent": intent, "expanded_query": expanded}
        except json.JSONDecodeError:
            pass

    plain_expanded = cleaned if cleaned and not cleaned.startswith("{") else fallback_query
    return {
        "intent": "global_synthesis" if _has_global_synthesis_signal(fallback_query) else "fact_search",
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
    rewrites the user's question into a self-contained query in a single LLM call.
    """
    target_model = model_name or EXPAND_MODEL

    messages = [{"role": "system", "content": QUERY_EXPANSION_PROMPT}]

    if chat_history:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in chat_history
        )
        user_prompt = (
            f"Conversation History:\n{history_text}\n\n"
            f"User's Latest Question: {query}"
        )
    else:
        user_prompt = f"User's Question: {query}"

    messages.append({"role": "user", "content": user_prompt})

    try:
        raw_output = invoke_llm_traced(
            model_name=target_model,
            messages=messages,
            timeout=30,
        ).strip()
        return _parse_router_response(raw_output, fallback_query=query)
    except Exception as e:
        print(f"[route_and_expand_query] Fallback due to error: {e}")
        return {
            "intent": "global_synthesis" if _has_global_synthesis_signal(query) else "fact_search",
            "expanded_query": query,
        }


def expand_query(
    query: str,
    chat_history: list[dict] | None = None,
    model_name: str | None = None,
) -> str:
    """Backward-compatible helper returning only the expanded query string."""
    return route_and_expand_query(query, chat_history, model_name)["expanded_query"]