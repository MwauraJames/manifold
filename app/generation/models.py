"""Dynamic Model Discovery & LiteLLM Configuration."""

import os
from pathlib import Path
from typing import Dict, Iterator, List
from dotenv import load_dotenv
import litellm
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# Load .env from project root or generation directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
load_dotenv(override=True)


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


# In app/generation/models.py:

PROVIDER_CATALOG: Dict[str, dict] = {
    "GROQ_API_KEY": {
        "provider": "Groq",
        "models": [
            {
                "id": "groq/openai/gpt-oss-120b",
                "name": "GPT-OSS 120B (Groq - Flagship Reasoning)",
            },
            {
                "id": "groq/openai/gpt-oss-20b",
                "name": "GPT-OSS 20B (Groq - Ultra-Fast ~1000 tok/s)",
            },
            {
                "id": "groq/qwen/qwen3.8-27b",
                "name": "Qwen 3.8 27B (Groq - Multimodal Vision & Reasoning)",
            },
            {
                "id": "groq/meta-models/muse-spark-open",
                "name": "Muse Spark Open 816B MoE (Groq - Reasoning)",
            },
            {
                "id": "groq/minimaxai/minimax-m2.7",
                "name": "MiniMax M2.7 (Groq - 196K Context)",
            },
            {
                "id": "groq/llama-3.3-70b-versatile",
                "name": "Llama 3.3 70B Enterprise (Groq)",
            },
            {
                "id": "groq/openai/gpt-oss-safeguard-20b",
                "name": "Safety GPT-OSS 20B (Groq - Guardrail/Safe Chat)",
            },
        ],
    },
    # ... keep ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, KIMI_API_KEY as-is
    "ANTHROPIC_API_KEY": {
        "provider": "Anthropic",
        "models": [
            {
                "id": "anthropic/claude-haiku-4-5",
                "name": "Claude 4.5 Haiku",
            },
            {
                "id": "anthropic/claude-sonnet-4-6",
                "name": "Claude 4.6 Sonnet",
            },
        ],
    },
    "OPENAI_API_KEY": {
        "provider": "OpenAI",
        "models": [
            {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 Mini"},
            {"id": "openai/gpt-4.1", "name": "GPT-4.1"},
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini"},
        ],
    },
    "GEMINI_API_KEY": {
        "provider": "Google Gemini",
        "models": [
            {"id": "gemini/gemini-3.8-flash", "name": "Gemini 3.8 Flash"},
            
        ],
    },
    "OPENROUTER_API_KEY": {
        "provider": "OpenRouter",
        "models": [
            {
                "id": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
                "name": "Nemotron 3 Ultra (OpenRouter)",
            },
        ],
    },
    "KIMI_API_KEY": {
        "provider": "Moonshot AI (Kimi)",
        "models": [
            {
                "id": "openai/moonshot-v1-8k",
                "name": "Moonshot v1 8K",
                "api_base": "https://api.moonshot.cn/v1",
                "api_key_env": "KIMI_API_KEY",
            }
        ],
    },
}

NON_LLM_ENV_KEYS = {
    "SUPABASE_KEY",
    "SUPABASE_API_KEY",
    "LLAMA_CLOUD_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "QDRANT_API_KEY",
    "COHERE_API_KEY",
}


def _auto_discover_unlisted_providers(seen_ids: set[str]) -> List[dict]:
    discovered: List[dict] = []
    models_by_provider = getattr(litellm, "models_by_provider", {})
    model_cost = getattr(litellm, "model_cost", {})

    for env_key, env_val in os.environ.items():
        if not env_key.endswith("_API_KEY"):
            continue
        if env_key in PROVIDER_CATALOG or env_key in NON_LLM_ENV_KEYS:
            continue

        val = env_val.strip()
        if not val or val.startswith("your_"):
            continue

        raw_prefix = env_key[: -len("_API_KEY")].lower()
        candidate_slugs = [
            raw_prefix,
            raw_prefix.replace("_", ""),
            f"{raw_prefix}_ai",
        ]

        matched_provider = next(
            (slug for slug in candidate_slugs if slug in models_by_provider),
            None,
        )
        if not matched_provider:
            continue

        provider_models = models_by_provider.get(matched_provider, [])
        added_for_provider = 0

        for m in provider_models:
            full_id = m if m.startswith(f"{matched_provider}/") else f"{matched_provider}/{m}"
            if full_id in seen_ids:
                continue

            info = model_cost.get(full_id) or model_cost.get(m) or {}
            mode = info.get("mode", "chat")
            if mode != "chat":
                continue

            lower_m = m.lower()
            if any(
                noise in lower_m
                for noise in ("embed", "moderation", "whisper", "tts", "pixtral", "-2024", "-240")
            ):
                continue

            short_name = full_id.split("/", 1)[-1]
            discovered.append(
                {
                    "id": full_id,
                    "name": f"{short_name} ({matched_provider.capitalize()})",
                    "provider": matched_provider.capitalize(),
                    "api_base": None,
                    "api_key_env": env_key,
                }
            )
            seen_ids.add(full_id)
            added_for_provider += 1

            if added_for_provider >= 8:
                break

    return discovered


def discover_available_models() -> List[dict]:
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    available: List[dict] = []
    seen_ids: set[str] = set()

    for env_var, info in PROVIDER_CATALOG.items():
        api_key = os.getenv(env_var, "").strip()
        if api_key and not api_key.startswith("your_"):
            for m in info["models"]:
                if m["id"] not in seen_ids:
                    available.append(
                        {
                            "id": m["id"],
                            "name": m["name"],
                            "provider": info["provider"],
                            "api_base": m.get("api_base"),
                            "api_key_env": m.get("api_key_env", env_var),
                        }
                    )
                    seen_ids.add(m["id"])

    available.extend(_auto_discover_unlisted_providers(seen_ids))

    extra_models = os.getenv("EXTRA_MODELS", "").strip()
    if extra_models:
        for raw_id in extra_models.split(","):
            clean_id = raw_id.strip()
            if clean_id and clean_id not in seen_ids:
                provider = clean_id.split("/")[0].capitalize() if "/" in clean_id else "Custom"
                available.append(
                    {
                        "id": clean_id,
                        "name": f"{clean_id}",
                        "provider": provider,
                        "api_base": None,
                        "api_key_env": None,
                    }
                )
                seen_ids.add(clean_id)

    ollama_models = os.getenv("OLLAMA_MODELS", "").strip()
    if ollama_models:
        for raw_name in ollama_models.split(","):
            clean_name = raw_name.strip()
            model_id = f"ollama/{clean_name}"
            if clean_name and model_id not in seen_ids:
                available.append(
                    {
                        "id": model_id,
                        "name": f"{clean_name} (Local Ollama)",
                        "provider": "Ollama",
                        "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
                        "api_key_env": None,
                    }
                )
                seen_ids.add(model_id)

    return available


def get_default_model() -> str:
    models = discover_available_models()
    if models:
        return models[0]["id"]
    return "groq/openai/gpt-oss-120b"


def resolve_model_config(requested_model: str | None = None) -> dict:
    available = discover_available_models()
    available_map = {m["id"]: m for m in available}

    target_id = (requested_model or "").strip() or get_default_model()

    if target_id in available_map:
        meta = available_map[target_id]
        kwargs = {"model": target_id}
        if meta.get("api_base"):
            kwargs["api_base"] = meta["api_base"]
        if meta.get("api_key_env") and meta["api_key_env"] == "KIMI_API_KEY":
            kwargs["api_key"] = os.getenv("KIMI_API_KEY")
        return kwargs

    if "/" in target_id:
        return {"model": target_id}

    valid_ids = ", ".join(available_map.keys())
    raise ValueError(
        f"Model '{target_id}' is not available or its API key is missing. "
        f"Available models: {valid_ids}"
    )

def get_vision_model_candidates() -> List[str]:
    """
    Returns an ordered list of vision-capable LiteLLM model IDs based on
    configured API keys in .env, starting with gemini/gemini-3.8-flash.
    """
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

    candidates: List[str] = []

    # 1. Explicit override in .env (e.g., VISION_MODEL=gemini/gemini-3.8-flash)
    custom_vision = os.getenv("VISION_MODEL", "").strip()
    if custom_vision:
        candidates.append(custom_vision)

    # 2. Provider-based vision models matching your active PROVIDER_CATALOG
    vision_catalog = [
        ("GEMINI_API_KEY", "gemini/gemini-3.8-flash"),
        #("GEMINI_API_KEY", "gemini/gemini-2.5-pro"),
        ("OPENAI_API_KEY", "openai/gpt-4.1-mini"),
        ("OPENAI_API_KEY", "openai/gpt-4o-mini"),
        ("ANTHROPIC_API_KEY", "anthropic/claude-haiku-4-5"),
        ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-6"),
    ]

    for env_key, model_id in vision_catalog:
        val = os.getenv(env_key, "").strip()
        if val and not val.startswith("your_") and model_id not in candidates:
            candidates.append(model_id)

    # Optional: If a specific Groq vision model is set in .env, include it
    groq_vision = os.getenv("GROQ_VISION_MODEL", "").strip()
    if groq_vision and os.getenv("GROQ_API_KEY", "").strip():
        full_groq = groq_vision if groq_vision.startswith("groq/") else f"groq/{groq_vision}"
        if full_groq not in candidates:
            candidates.append(full_groq)

    if not candidates:
        raise RuntimeError(
            "No vision-capable LLM API key found. Set GEMINI_API_KEY, "
            "OPENAI_API_KEY, or ANTHROPIC_API_KEY in app/.env."
        )

    return candidates