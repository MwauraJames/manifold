# app/generation/response.py
import logging
import os
import re
import threading
from typing import Iterator

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from configs.config import Settings
from generation.models import (
    get_default_model,
    invoke_llm_traced,
    stream_llm_traced,
)
from generation.prompts import CHAT_SUMMARY_PROMPT, RESPONSE_SYSTEM_PROMPT

logger = logging.getLogger("manifold.memory")

settings = Settings()
CHAT_HISTORY_LIMIT = getattr(settings, "chat_history_limit", 6)
CHAT_RECENT_KEEP = getattr(settings, "chat_recent_keep", 2)
SUMMARY_MODEL = getattr(settings, "summary_model", None)

SUMMARY_MSG_MAX_CHARS = 1500
SUMMARY_PREV_MAX_CHARS = 3000

# Matches inline citation brackets like [1] or [12] without stripping array indexing like arr[0]
_INLINE_CITE_RE = re.compile(r"(?<!\w)\[\d{1,2}\]")

_chat_locks: dict[str, threading.Lock] = {}
_chat_locks_guard = threading.Lock()


def _get_chat_lock(chat_id: str) -> threading.Lock:
    with _chat_locks_guard:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[chat_id] = lock
        return lock


def _has_provider_key(model_id: str) -> bool:
    """Checks whether the API key for a LiteLLM model prefix is configured."""
    if "/" not in model_id:
        return True

    provider = model_id.split("/", 1)[0].lower()
    if provider == "ollama":
        return True
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY", "").strip())
    if provider == "gemini":
        return bool(
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )
    if provider == "openai":
        if "moonshot" in model_id.lower():
            return bool(os.getenv("KIMI_API_KEY", "").strip())
        return bool(os.getenv("OPENAI_API_KEY", "").strip())
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    return True


def _get_summary_model_candidates(explicit_model: str | None = None) -> list[str]:
    """Ordered low-latency model fallback chain for rolling conversation summaries."""
    raw_candidates = [
        explicit_model,
        os.getenv("SUMMARY_MODEL", "").strip() or None,
        SUMMARY_MODEL,
        "groq/openai/gpt-oss-20b",
        "gemini/gemini-3.8-flash",
        "openai/gpt-4.1-mini",
        "anthropic/claude-haiku-4-5",
    ]
    candidates: list[str] = []
    for model_id in raw_candidates:
        if not model_id or model_id in candidates:
            continue
        if model_id == explicit_model or _has_provider_key(model_id):
            candidates.append(model_id)

    return candidates or [get_default_model()]


def _clean_text_for_summary(text: str, max_chars: int) -> str:
    """Strips inline [N] citation markers and caps length to prevent TPM/size spikes."""
    cleaned = _INLINE_CITE_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + "…"
    return cleaned


def _align_recent_messages(messages: list[dict]) -> list[dict]:
    """Ensures recent turns start on a 'user' message with clean role fields."""
    cleaned = [
        {
            "role": "user" if m.get("role") == "user" else "assistant",
            "content": (m.get("content") or "").strip(),
        }
        for m in messages
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def _format_memory_trace_inputs(inputs: dict) -> dict:
    """Keeps LangSmith trace inputs concise and free of raw MongoStore handles."""
    msgs = inputs.get("messages_to_summarize") or []
    return {
        "chat_id": inputs.get("chat_id"),
        "summary_model": inputs.get("summary_model"),
        "messages_to_fold": len(msgs),
        "recent_kept": inputs.get("recent_kept_count", 0),
        "has_previous_summary": bool(inputs.get("previous_summary")),
    }


def _format_summary_call_inputs(inputs: dict) -> dict:
    """Resolves the primary summary model for display in LangSmith inputs."""
    msgs = inputs.get("messages_to_summarize") or []
    resolved = _get_summary_model_candidates(inputs.get("model_name"))[0]
    return {
        "model_name": resolved,
        "messages_count": len(msgs),
        "has_previous_summary": bool(inputs.get("previous_summary")),
    }


@traceable(
    name="summarize_chat_history",
    run_type="chain",
    process_inputs=_format_summary_call_inputs,
)
def summarize_chat_history(
    messages_to_summarize: list[dict],
    previous_summary: str | None = None,
    model_name: str | None = None,
) -> str | None:
    """
    Folds `previous_summary` and `messages_to_summarize` into a concise updated summary.
    Returns `None` on failure or empty model output so callers never corrupt MongoDB.
    """
    transcript_lines: list[str] = []
    for msg in messages_to_summarize:
        role = (msg.get("role") or "user").upper()
        body = _clean_text_for_summary(msg.get("content") or "", SUMMARY_MSG_MAX_CHARS)
        if body:
            transcript_lines.append(f"{role}: {body}")

    if not transcript_lines:
        return None

    transcript = "\n".join(transcript_lines)

    if previous_summary and previous_summary.strip():
        clean_prev = _clean_text_for_summary(previous_summary, SUMMARY_PREV_MAX_CHARS)
        user_content = (
            f"Previous Summary:\n{clean_prev}\n\n"
            f"New Conversation Turns to Fold In:\n{transcript}"
        )
    else:
        user_content = f"Conversation Turns to Summarize:\n{transcript}"

    summary_messages = [
        {"role": "system", "content": CHAT_SUMMARY_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for candidate in _get_summary_model_candidates(model_name):
        try:
            output = invoke_llm_traced(
                model_name=candidate,
                messages=summary_messages,
                timeout=45,
            ).strip()
            output = _INLINE_CITE_RE.sub("", output).strip()
            if output:
                return output
            logger.warning(
                "[Memory:Summary] Model '%s' returned empty summary; trying next fallback.",
                candidate,
            )
        except Exception as exc:
            logger.warning(
                "[Memory:Summary] Model '%s' failed (%s); trying next fallback.",
                candidate,
                exc,
            )

    logger.error("[Memory:Summary] All summary model candidates failed; keeping cursor unchanged.")
    return None


@traceable(
    name="prepare_rolling_memory",
    run_type="chain",
    process_inputs=_format_memory_trace_inputs,
)
def _traced_memory_rollover(
    chat_id: str,
    summary_model: str,
    messages_to_summarize: list[dict],
    previous_summary: str | None,
    recent_kept_count: int,
) -> dict:
    """
    Traced execution boundary that only runs when a chat actually crosses
    `CHAT_HISTORY_LIMIT`, avoiding no-op traces on early turns.
    """
    rt = get_current_run_tree()
    if rt:
        rt.metadata.update(
            {
                "chat_id": chat_id,
                "session_id": chat_id,
                "summary_model": summary_model,
            }
        )

    new_summary = summarize_chat_history(
        messages_to_summarize=messages_to_summarize,
        previous_summary=previous_summary,
        model_name=summary_model,
    )
    return {
        "updated": bool(new_summary),
        "summary": new_summary,
        "folded_messages": len(messages_to_summarize) if new_summary else 0,
        "last_summarized_id": messages_to_summarize[-1].get("id") if new_summary else None,
    }


def get_rolling_memory(mongo, chat_id: str) -> tuple[str | None, list[dict]]:
    """
    Fast, non-blocking memory read for the start of a turn (zero LLM calls).
    Caps raw recent messages to `CHAT_HISTORY_LIMIT` if a prior background
    summarization is still catching up or failed.
    """
    state = mongo.get_chat_memory_state(chat_id)
    summary: str | None = state["summary"]
    unsummarized: list[dict] = state["unsummarized_messages"]

    if len(unsummarized) > CHAT_HISTORY_LIMIT:
        unsummarized = unsummarized[-CHAT_HISTORY_LIMIT:]

    return summary, _align_recent_messages(unsummarized)


def prepare_rolling_memory(
    mongo,
    chat_id: str,
    summary_model: str | None = None,
) -> tuple[str | None, list[dict]]:
    """
    Transactionally summarizes overflow turns for `chat_id` under a per-chat lock.
    Emits a LangSmith trace only when `len(unsummarized) > CHAT_HISTORY_LIMIT`.
    """
    lock = _get_chat_lock(chat_id)
    with lock:
        state = mongo.get_chat_memory_state(chat_id)
        current_summary: str | None = state["summary"]
        summarized_count: int = state["summarized_count"]
        unsummarized: list[dict] = state["unsummarized_messages"]

        if len(unsummarized) > CHAT_HISTORY_LIMIT:
            if CHAT_RECENT_KEEP > 0:
                to_summarize = unsummarized[:-CHAT_RECENT_KEEP]
                recent_to_keep = unsummarized[-CHAT_RECENT_KEEP:]
            else:
                to_summarize = unsummarized
                recent_to_keep = []

            if to_summarize:
                resolved_model = _get_summary_model_candidates(summary_model)[0]
                rollover = _traced_memory_rollover(
                    chat_id=chat_id,
                    summary_model=resolved_model,
                    messages_to_summarize=to_summarize,
                    previous_summary=current_summary,
                    recent_kept_count=len(recent_to_keep),
                )
                new_summary = rollover["summary"]
                if new_summary:
                    new_summarized_count = summarized_count + len(to_summarize)
                    last_id = rollover["last_summarized_id"]
                    mongo.update_chat_summary(
                        chat_id=chat_id,
                        new_summary=new_summary,
                        new_summarized_count=new_summarized_count,
                        last_summarized_id=last_id,
                    )
                    logger.info(
                        "[Memory:Summary] Updated chat_id=%s | model=%s | folded=%d | kept=%d | last_id=%s",
                        chat_id,
                        resolved_model,
                        len(to_summarize),
                        len(recent_to_keep),
                        last_id,
                    )
                    current_summary = new_summary
                    unsummarized = recent_to_keep
                else:
                    # Leave cursor untouched; cap raw messages returned to caller
                    unsummarized = unsummarized[-CHAT_HISTORY_LIMIT:]

        return current_summary, _align_recent_messages(unsummarized)


def schedule_rolling_memory_update(
    mongo,
    chat_id: str,
    summary_model: str | None = None,
) -> None:
    """
    Fires post-turn rolling summarization on a daemon thread so the next turn
    starts with an up-to-date summary without adding TTFT latency.
    """
    def _worker() -> None:
        try:
            prepare_rolling_memory(
                mongo=mongo,
                chat_id=chat_id,
                summary_model=summary_model,
            )
        except Exception as exc:
            logger.exception(
                "[Memory:Background] Failed rolling summary update for chat_id=%s: %s",
                chat_id,
                exc,
            )

    threading.Thread(
        target=_worker,
        name=f"manifold-summary-{chat_id[:8]}",
        daemon=True,
    ).start()


def build_messages(
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
    summary: str | None = None,
) -> list[dict]:
    """
    Builds a strict multi-provider-compatible message list:
      1. Single `system` message (RESPONSE_SYSTEM_PROMPT + optional rolling summary)
      2. Cleanly alternating `user` / `assistant` recent turns
      3. Final `user` turn containing Source Documents + Question
    """
    system_content = RESPONSE_SYSTEM_PROMPT
    if summary and summary.strip():
        system_content = (
            f"{RESPONSE_SYSTEM_PROMPT}\n\n"
            f"<conversation_summary>\n"
            f"{summary.strip()}\n"
            f"</conversation_summary>"
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]

    if chat_history:
        for msg in _align_recent_messages(chat_history):
            role = msg["role"]
            content = msg["content"]
            # Merge adjacent same-role turns so strict providers (Anthropic/Gemini) never 400
            if len(messages) > 1 and messages[-1]["role"] == role:
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
            else:
                messages.append({"role": role, "content": content})

    current_turn = (
        f"Source Documents:\n{context}\n\n"
        f"Question: {query}"
    )

    if len(messages) > 1 and messages[-1]["role"] == "user":
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{current_turn}"
    else:
        messages.append({"role": "user", "content": current_turn})

    return messages


@traceable(name="final_response", run_type="chain")
def final_response(
    model_name: str | None,
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
    summary: str | None = None,
) -> str:
    """Non-streaming completion via traced LiteLLM wrapper."""
    messages = build_messages(
        query=query,
        context=context,
        chat_history=chat_history,
        summary=summary,
    )
    return invoke_llm_traced(
        model_name=model_name,
        messages=messages,
        timeout=60,
    )


@traceable(name="stream_final_response", run_type="chain")
def stream_final_response(
    model_name: str | None,
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
    summary: str | None = None,
) -> Iterator[str]:
    """Streaming completion generator via traced LiteLLM wrapper."""
    messages = build_messages(
        query=query,
        context=context,
        chat_history=chat_history,
        summary=summary,
    )
    yield from stream_llm_traced(
        model_name=model_name,
        messages=messages,
        timeout=60,
    )