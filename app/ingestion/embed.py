# app/ingestion/embed.py
import os
from llama_index.embeddings.cohere import CohereEmbedding
from configs.config import Settings

settings = Settings()

class Embed:
    def get_embedding(self):
        api_key = os.getenv("COHERE_API_KEY") or getattr(settings, "cohere_api_key", None)
        model_name = os.getenv("EMBED_MODEL") or getattr(
            settings, "embed_model", "embed-english-v3.0"
        )
        return CohereEmbedding(
            api_key=api_key,
            model_name=model_name,
            input_type="search_document",
            embed_batch_size=96,  # Max allowed by Cohere (cuts API calls by ~9.6x)
            max_retries=10,       # Built-in exponential backoff on 429
        )