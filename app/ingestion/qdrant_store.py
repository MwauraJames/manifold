# app/ingestion/qdrant_store.py
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import qdrant_client
from qdrant_client.http import models as rest
from llama_index.vector_stores.qdrant import QdrantVectorStore

from configs.config import Settings

_APP_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_APP_ENV, override=True)
load_dotenv(override=True)

logger = logging.getLogger("manifold.qdrant")
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

        # Use HTTPS REST (prefer_grpc=False) to avoid port 6334 firewall/gRPC timeouts
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=False,
            timeout=60,
        )
        self._vector_store = None
        self._cached_index = None

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
        Checks existing payload schema and only creates missing keyword indexes.
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                return

            collection_info = self.client.get_collection(self.collection_name)
            existing_schema = collection_info.payload_schema or {}

            for field in FILTERABLE_FIELDS:
                if field not in existing_schema:
                    logger.info("[Qdrant] Creating keyword index for '%s'...", field)
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=rest.PayloadSchemaType.KEYWORD,
                    )
        except Exception as e:
            logger.warning("[Qdrant] Could not verify payload indexes at startup: %s", e)

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