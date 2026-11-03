# app/generation/response.py
from typing import Iterator
from langsmith import traceable
from generation.prompts import RESPONSE_SYSTEM_PROMPT, CHAT_SUMMARY_PROMPT
from generation.models import invoke_llm_traced, stream_llm_traced
from configs.config import Settings

settings = Settings()
CHAT_HISTORY_LIMIT = getattr(settings, "chat_history_limit", 6)
CHAT_RECENT_KEEP = getattr(settings, "chat_recent_keep", 2)
SUMMARY_MODEL = getattr(settings, "summary_model", None)


@traceable(name="summarize_chat_history", run_type="chain")
def summarize_chat_history(
    messages_to_summarize: list[dict],
    previous_summary: str | None = None,
    model_name: str | None = None,
) -> str:
    """
    Uses `settings.summary_model` (unless explicitly overridden) to fold
    `previous_summary` and `messages_to_summarize` into a single updated summary.
    """
    target_model = model_name or SUMMARY_MODEL

    transcript = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages_to_summarize
    )

    if previous_summary:
        user_content = (
            f"Previous Summary:\n{previous_summary}\n\n"
            f"New Conversation Turns to Fold In:\n{transcript}"
        )
    else:
        user_content = f"Conversation Turns to Summarize:\n{transcript}"

    summary_messages = [
        {"role": "system", "content": CHAT_SUMMARY_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        return invoke_llm_traced(
            model_name=target_model,
            messages=summary_messages,
            timeout=60,
        ).strip()
    except Exception as e:
        print(f"[summarize_chat_history] Error generating summary, keeping previous: {e}")
        return previous_summary or transcript


@traceable(name="prepare_rolling_memory", run_type="chain")
def prepare_rolling_memory(
    mongo,
    chat_id: str,
    summary_model: str | None = None,
) -> tuple[str | None, list[dict]]:
    """
    Evaluates rolling memory for `chat_id` and summarizes overflow turns
    using `settings.summary_model`.
    """
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
            new_summary = summarize_chat_history(
                messages_to_summarize=to_summarize,
                previous_summary=current_summary,
                model_name=summary_model or SUMMARY_MODEL,
            )
            new_summarized_count = summarized_count + len(to_summarize)
            mongo.update_chat_summary(
                chat_id=chat_id,
                new_summary=new_summary,
                new_summarized_count=new_summarized_count,
            )
            current_summary = new_summary
            unsummarized = recent_to_keep

    return current_summary, unsummarized


def build_messages(
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
    summary: str | None = None,
) -> list[dict]:
    """
    Builds an OpenAI/LiteLLM-compatible multi-turn message list:
      1. System prompt
      2. Rolling Summary of older conversation history (if present)
      3. Recent raw ('user' / 'assistant') turns
      4. Current ('user') turn containing the retrieved Source Documents + Question
    """
    messages = [{"role": "system", "content": RESPONSE_SYSTEM_PROMPT}]

    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"Summary of Earlier Conversation:\n{summary}",
            }
        )

    if chat_history:
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

    current_turn = (
        f"Source Documents:\n{context}\n\n"
        f"Question: {query}"
    )
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