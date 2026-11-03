from llama_index.embeddings.fastembed import FastEmbedEmbedding
from configs.config import Settings

settings = Settings()


class Embed:
    def get_embedding(self):
        return FastEmbedEmbedding(
            model_name=settings.embed_model,
            cache_dir="./models",
        )