from llama_index.embeddings.cohere import CohereEmbedding
from configs.config import Settings
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
settings = Settings()

class Embed:
    def get_embedding(self):
        return CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-english-v3.0",
            input_type="search_document",
        )