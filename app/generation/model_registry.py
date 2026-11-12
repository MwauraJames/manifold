"""Model registry: discover -> filter -> curate -> cache.

* discover : ask each provider (and local server) what the configured key / endpoint can use
* filter   : keep chat-capable models only, drop dated snapshots when an alias exists
* curate   : flag a short "featured" set so the UI stays small; everything else stays reachable
* cache    : per-source TTL, stale-while-revalidate, so requests never wait on the network
              (except the very first call, which is normally done at startup via warm_up()).
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from typing import Callable, Optional

import httpx
import litellm

logger = logging.getLogger("manifold.models")


# =============================================================================
# Config helpers (read lazily so .env loading order never matters)
# =============================================================================
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_list(name: str) -> list[str]:
    return [p.strip() for p in _env(name).split(",") if p.strip()]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = _env(name)
    return default if not raw else raw.lower() in ("1", "true", "yes", "on")


def _key(name: str) -> str:
    value = _env(name)
    return "" if value.startswith("your_") else value


def _gemini_key() -> str:
    return _key("GEMINI_API_KEY") or _key("GOOGLE_API_KEY")


DEFAULT_PROVIDER_ORDER = ["groq", "anthropic", "openai", "gemini", "kimi", "openrouter"]

# Last resort only: used when a provider is unreachable AND nothing was ever cached for it.
FALLBACK_MODELS = {
    "groq": ["groq/openai/gpt-oss-120b"],
    "openai": ["openai/gpt-4.1-mini"],
    "anthropic": ["anthropic/claude-haiku-4-5"],
    "gemini": ["gemini/gemini-3.8-flash"],
}

PROVIDER_LABELS = {
    "groq": "Groq",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
}

# OpenAI-compatible cloud APIs: slug -> (label, key env var, base-URL env var, default base URL)
# Adding another provider of this kind is one line.
OPENAI_COMPATIBLE_CLOUD = {
    "kimi": ("Moonshot AI (Kimi)", "KIMI_API_KEY", "KIMI_API_BASE", "https://api.moonshot.cn/v1"),
}


# =============================================================================
# Data model
# =============================================================================
@dataclass
class ModelInfo:
    id: str                          # what the UI sends back to /ask
    name: str
    provider: str                    # display label
    slug: str                        # grouping / ordering key
    source: str                      # "cloud" | "local" | "custom"
    litellm_model: str               # what litellm is actually called with
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None  # set only when the key must be passed explicitly
    vision: bool = False
    context_length: Optional[int] = None
    created: int = 0
    featured: bool = False

    def public(self) -> dict:
        """What the API exposes. Internal wiring (api_base, key env names) stays server-side."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "source": self.source,
            "vision": self.vision,
            "context_length": self.context_length,
            "featured": self.featured,
        }


class AuthError(Exception):
    """The provider rejected the API key."""


# =============================================================================
# HTTP + filtering helpers
# =============================================================================
def _get_json(url, headers=None, params=None, auth_codes=(401, 403)):
    r = httpx.get(url, headers=headers, params=params, timeout=float(_env("MODEL_DISCOVERY_TIMEOUT", "4")))
    if r.status_code in auth_codes:
        raise AuthError(f"HTTP {r.status_code} from {url}")
    r.raise_for_status()
    return r.json()


# Non-chat models that show up in provider listings. Heuristic: tune with MODEL_HIDE if needed.
_NOISE = re.compile(
    r"embed|moderation|whisper|tts|transcribe|audio|realtime|dall-e|image|imagen|veo|"
    r"rerank|guard|davinci|babbage|computer-use|aqa|search|live",
    re.I,
)
# Date / version suffixes: gpt-4.1-2025-04-14, ...-20250929, ...-0613, ...-001
_DATED = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{3,4})$")


def _drop_dated_snapshots(models: list[ModelInfo]) -> list[ModelInfo]:
    """Drop 'foo-2025-04-14' only when the plain alias 'foo' is also in the list."""
    present = {m.id for m in models}
    keep = []
    for m in models:
        match = _DATED.search(m.id)
        if match and m.id[: match.start()] in present:
            continue
        keep.append(m)
    return keep


def _is_openai_chat(model_id: str) -> bool:
    return bool(re.match(r"(gpt-|o\d|chatgpt-)", model_id)) and not re.search(r"instruct|codex", model_id)


def _supports_vision(litellm_model: str, hint: bool = False) -> bool:
    if hint:
        return True
    try:
        return bool(litellm.supports_vision(model=litellm_model))
    except Exception:
        return False


def _iso_to_epoch(value: Optional[str]) -> int:
    try:
        return int(datetime.fromisoformat((value or "").replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def _match(model_id: str, patterns: list[str]) -> bool:
    mid = model_id.lower()
    return any(fnmatch.fnmatchcase(mid, p.lower()) for p in patterns)


# =============================================================================
# Cloud fetchers (each returns normalized ModelInfo records)
# =============================================================================
def _fetch_openai_style(
    *, base, token, id_prefix, litellm_prefix, slug, label,
    accept=None, api_base=None, api_key_env=None,
) -> list[ModelInfo]:
    """Groq, OpenAI, Moonshot and most vendors share GET {base}/models with a Bearer token."""
    data = _get_json(f"{base}/models", {"Authorization": f"Bearer {token}"})
    out = []
    for m in data.get("data", []):
        raw = m.get("id") or ""
        if not raw or _NOISE.search(raw) or (accept and not accept(m)):
            continue
        litellm_model = f"{litellm_prefix}/{raw}"
        out.append(
            ModelInfo(
                id=f"{id_prefix}/{raw}",
                name=raw,
                provider=label,
                slug=slug,
                source="cloud",
                litellm_model=litellm_model,
                api_base=api_base,
                api_key_env=api_key_env,
                vision=_supports_vision(litellm_model, hint="vision" in raw.lower()),
                context_length=m.get("context_window") or m.get("context_length"),
                created=int(m.get("created") or 0),
            )
        )
    return out


def _fetch_anthropic() -> list[ModelInfo]:
    data = _get_json(
        "https://api.anthropic.com/v1/models",
        {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
        params={"limit": 1000},  # default page size is only 20
    )
    return [
        ModelInfo(
            id=f"anthropic/{m['id']}",
            name=m.get("display_name") or m["id"],
            provider="Anthropic",
            slug="anthropic",
            source="cloud",
            litellm_model=f"anthropic/{m['id']}",
            vision=True,
            created=_iso_to_epoch(m.get("created_at")),
        )
        for m in data.get("data", [])
        if m.get("id")
    ]


def _fetch_gemini() -> list[ModelInfo]:
    data = _get_json(
        "https://generativelanguage.googleapis.com/v1beta/models",
        {"x-goog-api-key": _gemini_key()},
        params={"pageSize": 1000},
        auth_codes=(400, 401, 403),  # Gemini answers 400 for an invalid key
    )
    out = []
    for m in data.get("models", []):
        raw = (m.get("name") or "").removeprefix("models/")
        if not raw.startswith("gemini") or _NOISE.search(raw):
            continue
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        out.append(
            ModelInfo(
                id=f"gemini/{raw}",
                name=m.get("displayName") or raw,
                provider="Google Gemini",
                slug="gemini",
                source="cloud",
                litellm_model=f"gemini/{raw}",
                vision=True,
                context_length=m.get("inputTokenLimit"),
            )
        )
    return out


def _fetch_openrouter() -> list[ModelInfo]:
    # Note: this endpoint is public, so it does not validate the key.
    data = _get_json(
        "https://openrouter.ai/api/v1/models",
        {"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}"},
    ).get("data", [])
    free_only = _env_flag("OPENROUTER_FREE_ONLY", True)
    out = []
    for m in data:
        arch = m.get("architecture") or {}
        if (arch.get("output_modalities") or ["text"]) != ["text"]:
            continue  # image / audio generators
        pricing = m.get("pricing") or {}
        is_free = m["id"].endswith(":free") or (
            str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
        )
        if free_only and not is_free:
            continue
        out.append(
            ModelInfo(
                id=f"openrouter/{m['id']}",
                name=m.get("name") or m["id"],
                provider="OpenRouter",
                slug="openrouter",
                source="cloud",
                litellm_model=f"openrouter/{m['id']}",
                vision="image" in (arch.get("input_modalities") or []),
                context_length=m.get("context_length"),
                created=int(m.get("created") or 0),
            )
        )
    return out


# =============================================================================
# Local fetchers (Ollama + any OpenAI-compatible local server)
# =============================================================================
_ollama_caps: dict[tuple[str, str], list[str]] = {}  # (name, digest) -> capabilities


def _ollama_capabilities(base: str, name: str, digest: str) -> Optional[list[str]]:
    """Newer Ollama versions report e.g. ['completion', 'vision', 'tools'] or ['embedding']."""
    cache_key = (name, digest)
    if cache_key in _ollama_caps:
        return _ollama_caps[cache_key]
    try:
        r = httpx.post(f"{base}/api/show", json={"model": name, "name": name}, timeout=float(_env("MODEL_DISCOVERY_TIMEOUT", "4")))
        r.raise_for_status()
        caps = r.json().get("capabilities")
    except Exception:
        caps = None
    if caps is not None:  # only cache real answers
        _ollama_caps[cache_key] = caps
    return caps


def _fetch_ollama() -> list[ModelInfo]:
    base = _env("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
    out = []
    for t in _get_json(f"{base}/api/tags").get("models", []):
        name = t.get("name") or t.get("model")
        if not name:
            continue
        caps = _ollama_capabilities(base, name, t.get("digest", ""))
        if caps is not None:
            if "completion" not in caps:
                continue  # embedding-only model
        else:  # older Ollama: no capabilities field, fall back to heuristics
            family = (t.get("details") or {}).get("family", "")
            if "embed" in name.lower() or family in ("bert", "nomic-bert"):
                continue
        out.append(
            ModelInfo(
                id=f"ollama/{name}",
                name=f"{name} (Ollama)",
                provider="Ollama",
                slug="ollama",
                source="local",
                litellm_model=f"ollama_chat/{name}",
                api_base=base,
                vision=bool(caps and "vision" in caps),
            )
        )
    return out


def _local_openai_endpoints() -> dict[str, str]:
    """LOCAL_OPENAI_ENDPOINTS=lmstudio=http://localhost:1234/v1,llamacpp=http://localhost:8080/v1"""
    endpoints = {}
    for item in _env_list("LOCAL_OPENAI_ENDPOINTS"):
        name, _, url = item.partition("=")
        if name.strip() and url.strip():
            endpoints[name.strip().lower()] = url.strip().rstrip("/")
    return endpoints


def _fetch_local_openai(name: str, base: str) -> list[ModelInfo]:
    """LM Studio, llama.cpp server, vLLM, ... all expose GET {base}/models."""
    data = _get_json(f"{base}/models")
    out = []
    for m in data.get("data", []):
        raw = m.get("id") or ""
        if not raw or _NOISE.search(raw):
            continue
        out.append(
            ModelInfo(
                id=f"{name}/{raw}",  # prefixed so two servers can't collide
                name=f"{raw} ({name})",
                provider=name.capitalize(),
                slug=name,
                source="local",
                litellm_model=f"openai/{raw}",
                api_base=base,
                vision=bool(re.search(r"vision|-vl|llava", raw, re.I)),
            )
        )
    return out


def _fetch_extra() -> list[ModelInfo]:
    """Manual escape hatch: EXTRA_MODELS=provider/model-id,provider/other-id"""
    return [
        ModelInfo(
            id=raw, name=raw, provider="Custom", slug="extra", source="custom", litellm_model=raw
        )
        for raw in _env_list("EXTRA_MODELS")
    ]


# =============================================================================
# Sources + cache
# =============================================================================
@dataclass
class Source:
    name: str
    kind: str  # "cloud" (slow-changing) | "local" (can come and go)
    fetch: Callable[[], list[ModelInfo]]

    @property
    def ttl(self) -> int:
        return int(_env("MODEL_CLOUD_TTL", "3600")) if self.kind == "cloud" else int(_env("MODEL_LOCAL_TTL", "30"))


def _sources() -> list[Source]:
    """Only sources that are actually configured. Rebuilt per call; it's cheap."""
    out: list[Source] = []
    if _key("GROQ_API_KEY"):
        out.append(Source("groq", "cloud", partial(
            _fetch_openai_style, base="https://api.groq.com/openai/v1", token=_key("GROQ_API_KEY"),
            id_prefix="groq", litellm_prefix="groq", slug="groq", label="Groq",
            accept=lambda m: m.get("active", True))))
    if _key("OPENAI_API_KEY"):
        out.append(Source("openai", "cloud", partial(
            _fetch_openai_style, base="https://api.openai.com/v1", token=_key("OPENAI_API_KEY"),
            id_prefix="openai", litellm_prefix="openai", slug="openai", label="OpenAI",
            accept=lambda m: _is_openai_chat(m.get("id", "")))))
    if _key("ANTHROPIC_API_KEY"):
        out.append(Source("anthropic", "cloud", _fetch_anthropic))
    if _gemini_key():
        out.append(Source("gemini", "cloud", _fetch_gemini))
    if _key("OPENROUTER_API_KEY"):
        out.append(Source("openrouter", "cloud", _fetch_openrouter))
    for slug, (label, key_env, base_env, default_base) in OPENAI_COMPATIBLE_CLOUD.items():
        if _key(key_env):
            base = _env(base_env, default_base).rstrip("/")
            out.append(Source(slug, "cloud", partial(
                _fetch_openai_style, base=base, token=_key(key_env), id_prefix=slug,
                litellm_prefix="openai", slug=slug, label=label, api_base=base, api_key_env=key_env)))

    # Local: always probed (fails fast and silently if nothing is listening).
    out.append(Source("ollama", "local", _fetch_ollama))
    for name, base in _local_openai_endpoints().items():
        out.append(Source(f"local:{name}", "local", partial(_fetch_local_openai, name, base)))
    if _env("EXTRA_MODELS"):
        out.append(Source("extra", "local", _fetch_extra))
    return out


@dataclass
class _SourceState:
    models: list[ModelInfo] = field(default_factory=list)
    fetched_at: float = 0.0
    ok: bool = False


_state: dict[str, _SourceState] = {}
_lock = threading.RLock()
_cold_lock = threading.Lock()
_bg_lock = threading.Lock()
_last_forced = 0.0


def _is_stale(src: Source) -> bool:
    st = _state.get(src.name)
    return st is None or (time.time() - st.fetched_at) > src.ttl


def _retry_soon(src: Source) -> float:
    """fetched_at value that makes a failed source stale again in ~60s (or its TTL if shorter)."""
    return time.time() - max(src.ttl - 60, 0)


def _refresh_source(src: Source) -> _SourceState:
    try:
        models = src.fetch()
        if src.kind == "cloud":
            models = _drop_dated_snapshots(models)
        return _SourceState(models, time.time(), True)
    except AuthError:
        logger.warning("[Models] %s rejected the API key; hiding this provider.", src.name)
        return _SourceState([], time.time(), False)
    except Exception as e:
        if src.kind == "local":  # a stopped local server simply has no models right now
            logger.debug("[Models] %s unreachable: %s", src.name, e)
            return _SourceState([], _retry_soon(src), False)
        prev = _state.get(src.name)
        logger.warning("[Models] %s discovery failed (%s); using %s.", src.name, e,
                       "last good list" if prev and prev.models else "fallback list")
        models = prev.models if prev and prev.models else _fallback(src.name)
        return _SourceState(models, _retry_soon(src), False)


def _fallback(slug: str) -> list[ModelInfo]:
    return [
        ModelInfo(id=i, name=i.split("/", 1)[1], provider=PROVIDER_LABELS.get(slug, slug.title()),
                  slug=slug, source="cloud", litellm_model=i)
        for i in FALLBACK_MODELS.get(slug, [])
    ]


def refresh(force: bool = False) -> None:
    """Refresh stale (or all, if force) sources in parallel, so total time = slowest source."""
    if _key("GOOGLE_API_KEY") and not _key("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = _key("GOOGLE_API_KEY")  # litellm reads GEMINI_API_KEY
    sources = _sources()
    todo = [s for s in sources if force or _is_stale(s)]
    results: dict[str, _SourceState] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            for src, st in zip(todo, pool.map(_refresh_source, todo)):
                results[src.name] = st
    with _lock:
        _state.update(results)
        active = {s.name for s in sources}
        for name in [n for n in _state if n not in active]:
            del _state[name]


def _refresh_in_background() -> None:
    if not _bg_lock.acquire(blocking=False):
        return  # one refresh at a time

    def run():
        try:
            refresh()
        finally:
            _bg_lock.release()

    threading.Thread(target=run, daemon=True, name="model-refresh").start()


def _snapshot() -> list[ModelInfo]:
    sources = _sources()
    if any(s.name not in _state for s in sources):
        with _cold_lock:  # first call only; waiters find nothing stale and return at once
            refresh()
    elif any(_is_stale(s) for s in sources):
        _refresh_in_background()  # serve the cached list now, update it behind the scenes
    with _lock:
        models = [m for s in sources for m in _state.get(s.name, _SourceState()).models]
    return _curate(models)


# =============================================================================
# Curation: keep the default UI short, keep everything reachable
# =============================================================================
def _provider_rank(slug: str) -> int:
    order = _env_list("MODEL_PROVIDER_ORDER") or DEFAULT_PROVIDER_ORDER
    return order.index(slug) if slug in order else len(order)


def _curate(models: list[ModelInfo]) -> list[ModelInfo]:
    hide = _env_list("MODEL_HIDE")
    patterns = _env_list("MODEL_FEATURED")
    per_provider = int(_env("MODEL_FEATURED_PER_PROVIDER", "3") or 3)

    seen: set[str] = set()
    visible: list[ModelInfo] = []
    for m in models:
        if m.id in seen or _match(m.id, hide):
            continue
        seen.add(m.id)
        visible.append(m)

    groups: dict[str, list[ModelInfo]] = {}
    for m in visible:
        groups.setdefault(m.slug, []).append(m)

    featured_ids: set[str] = set()
    for group in groups.values():
        if group[0].source != "cloud":  # local / custom: the user chose these, show them all
            featured_ids.update(m.id for m in group)
            continue
        picked = [m for m in group if _match(m.id, patterns)]
        if not picked:  # no explicit pattern hit for this provider -> newest N
            picked = sorted(group, key=lambda m: (m.created, m.id), reverse=True)[:per_provider]
        featured_ids.update(m.id for m in picked)

    out = [replace(m, featured=m.id in featured_ids) for m in visible]
    out.sort(key=lambda m: (_provider_rank(m.slug), m.slug, not m.featured, -m.created, m.name.lower()))
    return out


def _pick_default(models: list[ModelInfo]) -> Optional[str]:
    ids = {m.id for m in models}
    wanted = _env("MODEL_DEFAULT")
    if wanted:
        if wanted in ids:
            return wanted
        logger.warning("[Models] MODEL_DEFAULT=%s is not available; picking another.", wanted)
    for m in models:
        if m.featured:
            return m.id
    return models[0].id if models else None


# =============================================================================
# Public API (same names the rest of the app already imports)
# =============================================================================
def warm_up() -> None:
    """Call once at startup so the first /models or /ask never waits on the network."""
    models = _snapshot()
    logger.info("[Models] Discovered %d models (%d featured) from %d sources.",
                len(models), sum(m.featured for m in models), len(_state))


def discover_available_models() -> list[dict]:
    return [m.public() for m in _snapshot()]


def get_default_model() -> Optional[str]:
    return _pick_default(_snapshot())


def refresh_models() -> list[dict]:
    """Force a re-scan (e.g. after the user starts Ollama). Rate-limited to protect provider quotas."""
    global _last_forced
    now = time.time()
    if now - _last_forced >= 10:
        _last_forced = now
        refresh(force=True)
    return discover_available_models()


def resolve_model_config(requested_model: Optional[str] = None) -> dict:
    """Translate a registry id into the kwargs litellm needs. Raises ValueError if unavailable."""
    models = _snapshot()
    by_id = {m.id: m for m in models}
    target = (requested_model or "").strip() or (_pick_default(models) or "")

    m = by_id.get(target)
    if m:
        kwargs: dict = {"model": m.litellm_model}
        if m.api_base:
            kwargs["api_base"] = m.api_base
        if m.api_key_env:
            kwargs["api_key"] = os.getenv(m.api_key_env)
        elif m.source == "local" and m.litellm_model.startswith("openai/"):
            kwargs["api_key"] = _env("LOCAL_OPENAI_API_KEY", "local")  # SDK requires *some* key
        return kwargs

    # Off by default: otherwise any logged-in user could spend your keys on any model string.
    if _env_flag("ALLOW_ARBITRARY_MODELS") and "/" in target:
        return {"model": target}

    if not models:
        raise ValueError(
            "No models are available. Set a provider API key in .env or start a local server "
            "(Ollama, LM Studio, ...)."
        )
    raise ValueError(f"Model '{target}' is not available. Call GET /models for the current list.")


def resolve_model_id(requested_model: Optional[str] = None) -> str:
    """Validate a requested model and return its *registry id* (what the UI sees).

    Use this at the API edge and pass the id downstream. resolve_model_config() is then
    called once, at the actual LLM call. Resolving twice breaks for any model whose
    registry id differs from its litellm name (Ollama, Kimi, local servers).
    Raises ValueError if the model is unavailable.
    """
    resolve_model_config(requested_model)  # validation only
    return (requested_model or "").strip() or (_pick_default(_snapshot()) or "")


def get_vision_model_candidates(limit: int = 6) -> list[str]:
    """Ordered litellm model ids for vision calls: explicit VISION_MODEL, then discovered
    cloud models that accept images (featured + cheap-looking first)."""
    out: list[str] = []
    custom = _env("VISION_MODEL")
    if custom:
        out.append(custom)

    prefer = _env_list("VISION_PREFER") or ["*flash*", "*mini*", "*haiku*"]
    # Only models litellm can call from a bare id: no custom api_base / explicit key needed.
    vision = [
        m for m in _snapshot()
        if m.source == "cloud" and m.vision and not m.api_base and not m.api_key_env
    ]
    # sort is stable, so provider order is preserved inside each bucket
    vision.sort(key=lambda m: (not m.featured, not _match(m.id, prefer)))
    for m in vision:
        if m.litellm_model not in out:
            out.append(m.litellm_model)

    if not out:
        raise RuntimeError(
            "No vision-capable model found. Set VISION_MODEL, or add an API key for a provider "
            "with vision models (Gemini, OpenAI, Anthropic, ...)."
        )
    return out[:limit]