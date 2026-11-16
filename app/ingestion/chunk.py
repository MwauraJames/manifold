# app/ingestion/chunk.py
import re
from typing import Optional
from langsmith import traceable
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

from configs.config import Settings

settings = Settings()
CHUNK_SIZES = settings.chunk_sizes

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,4})\s+(.+)$", re.MULTILINE)


def _extract_section_header(text: str) -> Optional[str]:
    """Extracts the first Markdown heading from a chunk to serve as a readable section label."""
    if not text:
        return None
    match = _HEADING_RE.search(text)
    if match:
        heading = match.group(2).strip().strip("#").strip()
        return heading[:90]
    return None


@traceable
def chunk_documents(
    storage,
    embed_model,
    docs,
    user_id: str,
    subject: str,
    doc_id: str,
    filename: Optional[str] = None,
):
    # 1. Hierarchical split: 512 (parent) -> 128 (leaf) tokens.
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    all_nodes = node_parser.get_nodes_from_documents(docs)
    leaf_nodes = get_leaf_nodes(all_nodes)

    # 2. Build an in-memory lookup of all nodes so leaves can grab their parent's text.
    node_ById = {n.node_id: n for n in all_nodes}

    for leaf in leaf_nodes:
        parent_id = leaf.parent_node.node_id if leaf.parent_node else leaf.node_id
        parent_node = node_ById.get(parent_id, leaf)
        parent_text = parent_node.get_content()

        section = (
            _extract_section_header(parent_text)
            or _extract_section_header(leaf.get_content())
        )

        leaf.metadata.update({
            "user_id": user_id,
            "subject": subject,
            "doc_id": doc_id,
            "filename": filename or doc_id,
            "section": section or "",
            "parent_id": parent_id,
            "parent_text": parent_text,
        })

        # CRITICAL: Exclude payload fields from being concatenated into the
        # 128-token leaf text when computing dense & sparse embeddings!
        excluded_keys = [
            "parent_text",
            "parent_id",
            "user_id",
            "subject",
            "doc_id",
            "filename",
            "section",
        ]
        leaf.excluded_embed_metadata_keys.extend(excluded_keys)
        leaf.excluded_llm_metadata_keys.extend(excluded_keys)

    # 3. Index leaves into Qdrant (computes both dense and sparse vectors automatically).
    storage_context = StorageContext.from_defaults(vector_store=storage.vector_store())
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 4. Ensure fast keyword indexes exist on user_id, subject, doc_id, and parent_id.
    storage.ensure_payload_indexes()

    return {"nodes": len(all_nodes), "leaves": len(leaf_nodes), "doc_id": doc_id}