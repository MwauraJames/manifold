import qdrant_client
from qdrant_client.http import models as rest
from llama_index.vector_stores.qdrant import QdrantVectorStore

from configs.config import Settings

settings = Settings()

FILTERABLE_FIELDS = ("user_id", "subject", "doc_id", "parent_id")


class Storage:
    def __init__(self):
        # Single Qdrant client for the entire process
        self.client = qdrant_client.QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.collection_name = settings.qdrant_collection
        self._vector_store = None

    def vector_store(self) -> QdrantVectorStore:
        """
        Returns a QdrantVectorStore configured for native hybrid search
        (dense embeddings + sparse BM25/FastEmbed vectors).
        """
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                enable_hybrid=True,
                fastembed_sparse_model=settings.sparse_model,
            )
        return self._vector_store

    def ensure_payload_indexes(self):
        """
        Create keyword payload indexes on multi-tenant filter fields.
        Safe to call repeatedly after collection creation.
        """
        if not self.client.collection_exists(self.collection_name):
            return

        for field in FILTERABLE_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )

    def count_points(self, doc_id: str | None = None) -> int:
        """Count total points in collection, or points matching a specific doc_id."""
        if not self.client.collection_exists(self.collection_name):
            return 0

        query_filter = None
        if doc_id:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id",
                        match=rest.MatchValue(value=doc_id),
                    )
                ]
            )
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        ).count