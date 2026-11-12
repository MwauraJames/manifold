"""LiteLLM call wrappers (LangSmith-traced).

Model discovery, curation and caching live in generation/model_registry.py;
the public functions are re-exported here so existing imports keep working.
"""

from pathlib import Path
from typing import Iterator
from dotenv import load_dotenv
import litellm
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# Load .env from project root or generation directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
load_dotenv(override=True)

from generation.model_registry import (  # noqa: E402,F401  (re-exported for callers)
    discover_available_models,
    get_default_model,
    get_vision_model_candidates,
    refresh_models,
    resolve_model_config,
    resolve_model_id,
    warm_up,
)


def _parse_provider_and_model(full_model: str) -> tuple[str, str]:
    if "/" in full_model:
        provider, model_id = full_model.split("/", 1)
        return provider, model_id
    return "openai", full_model


def invoke_llm_traced(
    model_name: str | None,
    messages: list[dict],
    timeout: int = 60,
) -> str:
    """
    Wraps litellm.completion in a native LangSmith `llm` span so the UI shows:
      - Yellow AI model icon + dynamic model name (e.g. 'openai/gpt-oss-120b')
      - Token usage badge (input + output tokens)
      - Clean System / User / AI message bubbles
    """
    llm_kwargs = resolve_model_config(model_name)
    full_model = llm_kwargs["model"]
    provider, short_model = _parse_provider_and_model(full_model)

    @traceable(
        run_type="llm",
        name=short_model,
        metadata={"ls_provider": provider, "ls_model_name": short_model},
    )
    def _call(messages: list[dict]) -> dict:
        response = litellm.completion(
            **llm_kwargs,
            messages=messages,
            timeout=timeout,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)

        rt = get_current_run_tree()
        if rt and usage:
            rt.set(
                usage_metadata={
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
            )

        return {"role": "assistant", "content": content}

    result = _call(messages=messages)
    return result["content"]


def stream_llm_traced(
    model_name: str | None,
    messages: list[dict],
    timeout: int = 60,
) -> Iterator[str]:
    """
    Streaming equivalent with `run_type="llm"`, dynamic model badge, and token counting.
    """
    llm_kwargs = resolve_model_config(model_name)
    full_model = llm_kwargs["model"]
    provider, short_model = _parse_provider_and_model(full_model)

    @traceable(
        run_type="llm",
        name=short_model,
        metadata={"ls_provider": provider, "ls_model_name": short_model},
        reduce_fn=lambda chunks: {
            "role": "assistant",
            "content": "".join(chunks),
        },
    )
    def _stream(messages: list[dict]) -> Iterator[str]:
        response_stream = litellm.completion(
            **llm_kwargs,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            timeout=timeout,
        )
        collected: list[str] = []
        usage_data = None

        for chunk in response_stream:
            if getattr(chunk, "usage", None):
                usage_data = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                collected.append(token)
                yield token

        rt = get_current_run_tree()
        if rt:
            if usage_data:
                in_tok = getattr(usage_data, "prompt_tokens", 0) or 0
                out_tok = getattr(usage_data, "completion_tokens", 0) or 0
                tot_tok = getattr(usage_data, "total_tokens", in_tok + out_tok)
            else:
                try:
                    in_tok = litellm.token_counter(model=full_model, messages=messages)
                    out_tok = litellm.token_counter(
                        model=full_model, text="".join(collected)
                    )
                    tot_tok = in_tok + out_tok
                except Exception:
                    in_tok = out_tok = tot_tok = 0

            rt.set(
                usage_metadata={
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                }
            )

    return _stream(messages=messages)