from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    top_k: int = 50
    merge_ratio: float = 0.5       # Merge when > 50% of a parent's leaves are retrieved
    min_merge_leaves: int = 2
    chunk_sizes: list[int] = [512, 128]

    upload_dir: str = "/tmp/upload_temp"

    # Qdrant Configuration (defaults to local Docker/server instance; can also use path="./qdrant_storage")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "university_nodes"
    sparse_model: str = "Qdrant/bm25"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "study_assistant"

    # Utility LLM Models (passed to resolve_model_config)
    expand_model: str = "groq/openai/gpt-oss-120b"
    summary_model: str = "groq/openai/gpt-oss-120b"

    # Rolling Chat Memory Configuration
    chat_history_limit: int = 6   # Trigger summarization when unsummarized messages exceed this
    chat_recent_keep: int = 2     # Number of most recent messages to keep raw (un-summarized)

    temparature: float = 0.2

    rerank_model: str = "embed-english-v3.0"
    rerank_top_n: int = 10
    rerank_threshold: float = 0.68
    leaf_score_threshold: float = 0.00

    # Maximum characters (~4 chars per token) sampled across documents for global_synthesis queries
    synthesis_max_chars: int = 8000

    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_model_cache_dir: str = "./models"

    supabase_url: str
    supabase_key: str

settings = Settings()