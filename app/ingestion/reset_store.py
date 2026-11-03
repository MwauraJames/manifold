from qdrant_client.http import models as rest
# Add to app/ingestion/reset_store.py
from qdrant_client.models import Filter, FieldCondition, MatchValue


def delete_by_doc_id(storage, doc_id: str):
    """Delete all points belonging to a single document without affecting other users/docs."""
    if not storage.client.collection_exists(storage.collection_name):
        return

    storage.client.delete(
        collection_name=storage.collection_name,
        points_selector=rest.FilterSelector(
            filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id",
                        match=rest.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )


def reset_all(storage):
    """Wipe the entire Qdrant collection (use only for dev resets, not per-upload)."""
    if storage.client.collection_exists(storage.collection_name):
        storage.client.delete_collection(storage.collection_name)
    storage._vector_store = None

    

def delete_by_subject(storage, user_id: str, subject: str, doc_ids: list[str] | None = None):
    """
    Deletes all Qdrant vector/sparse points belonging to (user_id, subject).
    Also deletes by doc_ids to guarantee cleanup regardless of payload nesting.
    """
    if doc_ids:
        for doc_id in doc_ids:
            delete_by_doc_id(storage, doc_id)

    try:
        storage.client.delete(
            collection_name=storage.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="subject", match=MatchValue(value=subject)),
                ]
            ),
        )
    except Exception:
        pass





    # 4. Rerank against the expanded/contextualized query so pronouns don't confuse the reranker!
        
    