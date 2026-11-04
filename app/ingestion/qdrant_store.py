# app/ingestion/storage.py
import os
from dotenv import load_dotenv
import qdrant_client
from qdrant_client.http import models as rest
from llama_index.vector_stores.qdrant import QdrantVectorStore

from configs.config import Settings

from pathlib import Path
from dotenv import load_dotenv

_APP_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_APP_ENV, override=True)
load_dotenv(override=True)

settings = Settings()
FILTERABLE_FIELDS = ("user_id", "subject", "doc_id", "parent_id")


class Storage:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL") or getattr(
            settings, "qdrant_url", "http://localhost:6333"
        )
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or getattr(
            settings, "qdrant_api_key", None
        )
        if qdrant_api_key and (
            not qdrant_api_key.strip() or qdrant_api_key.startswith("your_")
        ):
            qdrant_api_key = None

        self.collection_name = os.getenv("QDRANT_COLLECTION") or getattr(
            settings, "qdrant_collection", "university_nodes"
        )

        # Single Qdrant client for the entire process
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )
        self._vector_store = None

        # Ensure keyword indexes exist on Qdrant Cloud if collection already exists
        self.ensure_payload_indexes()

    def vector_store(self) -> QdrantVectorStore:
        """
        Returns a QdrantVectorStore configured for native hybrid search
        (dense embeddings + sparse BM25/FastEmbed vectors).
        """
        if self._vector_store is None:
            sparse_model = os.getenv("SPARSE_MODEL") or getattr(
                settings, "sparse_model", "Qdrant/bm25"
            )
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                enable_hybrid=True,
                fastembed_sparse_model=sparse_model,
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