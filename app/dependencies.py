# app/dependencies.py
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from llama_cloud import LlamaCloud
import cohere

from configs.config import Settings
from ingestion.qdrant_store import Storage
from ingestion.mongo_store import MongoStore
from ingestion.embed import Embed
from schemas import IngestionJob

load_dotenv()

# ---------- Logging Configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("manifold.main")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _preview(text: Optional[str], max_len: int = 120) -> str:
    """Returns a single-line truncated preview of text for readable terminal logs."""
    if not text:
        return "<empty>"
    cleaned = " ".join(text.strip().split())
    return cleaned if len(cleaned) <= max_len else f"{cleaned[:max_len]}..."


def safe_slug(value: str) -> str:
    """
    Replaces slashes, backslashes, spaces, and unsafe path characters with '_'
    so subject names like 'CS 101/AI' never break file paths or URLs.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


# ---------- Shared State & Queues ----------
settings = Settings()

ingestion_queue: asyncio.Queue[IngestionJob] = asyncio.Queue()
active_streams: dict[str, bool] = {}

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

# ---------- Clients & Warmup ----------
logger.info("[Init] Initializing Qdrant Storage, MongoStore, Embed model, LlamaCloud, and Cohere...")
storage = Storage()          # Qdrant (Vectors + Hybrid Search)
mongo = MongoStore()         # MongoDB (Raw PDFs/Images, Parsed Docs, Chat History, Subjects)
embed_model = Embed().get_embedding()

try:
    embed_model.get_query_embedding("warmup")
    vs = storage.vector_store()
    if hasattr(vs, "_sparse_doc_fn") and vs._sparse_doc_fn is not None:
        vs._sparse_doc_fn(["warmup"])
    logger.info("[Init] Dense + Sparse embedding models and QdrantVectorStore warmed up.")
except Exception as e:
    logger.warning("[Init] Warmup warning: %s", e)

client = LlamaCloud()
co = cohere.ClientV2()
logger.info("[Init] All core clients initialized successfully.")