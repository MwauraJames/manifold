---
title: Manifold API
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
startup_duration_timeout: 30m
---

# 🌌 Manifold — Adaptive Hybrid-RAG & Synthesis Engine

**Manifold** is a production-grade Adaptive Retrieval-Augmented Generation (RAG) backend built with FastAPI, Qdrant, MongoDB, and LiteLLM[cite: 1, 9, 12]. It ingests complex academic PDFs (preserving LaTeX equations and code blocks) into hierarchical parent-child vector spaces and dynamically routes queries between precision hybrid vector search and document-wide stratified synthesis.

---

## ✨ Key Architectural Highlights

* **Adaptive Intent Routing (Zero Extra Latency)**: A single-pass pre-retrieval router classifies queries into `fact_search` (specific concepts, formulas, protocols) or `global_synthesis` (exam question generation, flashcards, full-unit summaries) while simultaneously rewriting conversational pronouns into standalone search queries[cite: 3, 6].
* **Hierarchical Hybrid Search & Stateless Parent Promotion**: Indexes 128-token leaf chunks in Qdrant using both dense embeddings (`BAAI/bge-large-en-v1.5` via ONNX FastEmbed) and sparse BM25 representations (`Qdrant/bm25`)[cite: 7, 9]. Surviving sibling leaves above `leaf_score_threshold` automatically promote into 512-token parent contexts[cite: 7].
* **Cross-Encoder Precision Guardrail**: Candidate passages are scored by Cohere `rerank-v4.0-pro` with a strict `0.70` relevance cutoff to eliminate hallucinations on factual lookups.
* **Stratified Synthesis Sampling**: Meta-queries (`global_synthesis`) bypass the pointwise cross-encoder cutoff and sample evenly spaced structural blocks across all ready documents in MongoDB (`parsed_markdown`) within a strict token budget[cite: 5, 14].
* **Rolling Window Memory Summarization**: Tracks `summary` and `summarized_count` per chat session in MongoDB, automatically folding older turns into a dense academic summary while preserving raw message history for the UI[cite: 4, 5].
* **3-Tier Dynamic Model Discovery**: Automatically discovers configured providers (`Groq`, `OpenAI`, `Anthropic`, `Gemini`, `OpenRouter`, `Kimi`, `Ollama`, plus any `<PROVIDER>_API_KEY` supported by LiteLLM) and streams tokens via Server-Sent Events (SSE) with Mid-Stream Stop and Edit-Last-Prompt controls[cite: 1, 2].
* **Full LangSmith Observability**: Custom `@traceable(run_type="llm")` spans track token usage, provider badges, routing tags (`intent:fact_search` / `intent:global_synthesis`), and group multi-turn sessions via `session_id` in the LangSmith Threads view.

---

## 🏗️ System Architecture

```text
User Query + Chat History
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  1. Rolling Memory Manager (prepare_rolling_memory)    │
│     Folds overflow turns (>6) into MongoDB summary     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  2. Adaptive Router & Query Rewriter (expand.py)       │
│     Outputs JSON: { intent, expanded_query }           │
└───────────┬──────────────────────────────┬─────────────┘
            │                              │
  [intent: fact_search]          [intent: global_synthesis]
            │                              │
            ▼                              ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│ Qdrant Hybrid Retriever   │  │ MongoDB Stratified      │
│ Dense (BGE) + Sparse BM25 │  │ Markdown Sampler        │
│            │              │  │            │            │
│ Stateless Parent Merge    │  │ Evenly-spaced sections  │
│ (128 leaves -> 512 parent)│  │ across Subject / Doc    │
│            │              │  └────────────┬────────────┘
│ Cohere Rerank v4.0 Pro    │               │
│ (Threshold >= 0.70)       │               │
└───────────┬───────────────┘               │
            └──────────────┬────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  3. LiteLLM Traced Generator (SSE Stream / JSON)       │
│     Supports /stop interruption & /edit_last rollback  │
└────────────────────────────────────────────────────────┘