# app/retrieval/hybrid_search.py
from typing import Optional
from collections import Counter
from langsmith import traceable
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from configs.config import Settings

settings = Settings()
# Cap per-modality leaf fetch at 14 so Qdrant doesn't return 50 leaves per query
TOP_K = min(getattr(settings, "top_k", 14), 14)
MERGE_RATIO = getattr(settings, "merge_ratio", 0.5)
MIN_MERGE_LEAVES = getattr(settings, "min_merge_leaves", 2)
LEAF_SCORE_THRESHOLD = settings.leaf_score_threshold
MAX_RERANK_CANDIDATES = getattr(settings, "max_rerank_candidates", 10)


def build_metadata_filters(
    user_id: str,
    subject: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> MetadataFilters:
    """Build Qdrant payload filters for user_id, and optionally subject and doc_id."""
    filters = [
        MetadataFilter(key="user_id", value=user_id, operator=FilterOperator.EQ)
    ]
    if subject:
        filters.append(
            MetadataFilter(key="subject", value=subject, operator=FilterOperator.EQ)
        )
    if doc_id:
        filters.append(
            MetadataFilter(key="doc_id", value=doc_id, operator=FilterOperator.EQ)
        )
    return MetadataFilters(filters=filters)


def build_hybrid_retriever(
    storage,
    embed_model,
    user_id: str,
    subject: Optional[str] = None,
    doc_id: Optional[str] = None,
):
    """
    Native Qdrant Hybrid Search:
      1. Caches VectorStoreIndex across requests to avoid rebuild overhead.
      2. Filters collection by user_id (+ optional subject / doc_id) in Qdrant payload.
      3. Runs dense + sparse (BM25) search over 128-token leaves.
    """
    if getattr(storage, "_cached_index", None) is None:
        storage._cached_index = VectorStoreIndex.from_vector_store(
            storage.vector_store(),
            embed_model=embed_model,
        )

    filters = build_metadata_filters(user_id=user_id, subject=subject, doc_id=doc_id)

    return storage._cached_index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=TOP_K,
        sparse_top_k=TOP_K,
        filters=filters,
    )


@traceable(run_type="retriever")
def merge_leaves_to_parents(
    results,
    max_candidates: int = MAX_RERANK_CANDIDATES,
) -> list[str]:
    """
    1. Filters out low-scoring leaves below LEAF_SCORE_THRESHOLD (0.0 - 1.0 scale).
    2. Counts how many surviving leaves share each `parent_id`.
    3. If a parent has >= MIN_MERGE_LEAVES surviving leaves, merges them into
       a single 512-token `parent_text` for the reranker.
    4. Caps returned candidates at `max_candidates` (default 10) to keep
       Cohere reranking fast (<0.35s).
    """
    if not results:
        return []

    qualified_leaves = [
        r for r in results
        if (r.score is None or r.score >= LEAF_SCORE_THRESHOLD)
    ]

    if not qualified_leaves:
        return []

    parent_counts = Counter()
    for r in qualified_leaves:
        meta = r.node.metadata or {}
        pid = meta.get("parent_id", r.node.node_id)
        parent_counts[pid] += 1

    parents_to_merge = {
        pid for pid, count in parent_counts.items()
        if count >= MIN_MERGE_LEAVES
    }

    merged_parent_ids = set()
    candidate_texts = []

    for r in qualified_leaves:
        meta = r.node.metadata or {}
        pid = meta.get("parent_id", r.node.node_id)

        if pid in parents_to_merge:
            if pid not in merged_parent_ids:
                merged_parent_ids.add(pid)
                parent_text = meta.get("parent_text") or r.node.get_content()
                candidate_texts.append(parent_text)
        else:
            candidate_texts.append(r.node.get_content())

        if len(candidate_texts) >= max_candidates:
            break

    return candidate_texts