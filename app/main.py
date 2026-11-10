# app/main.py
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dependencies import logger, settings
from pipelines.worker import ingestion_worker
from routers import auth, chats, documents, subjects


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the sequential ingestion worker when FastAPI boots up."""
    logger.info("[Lifespan] Booting Manifold FastAPI server...")
    worker_task = asyncio.create_task(ingestion_worker())
    yield
    logger.info("[Lifespan] Shutting down background ingestion worker...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("[Lifespan] Background ingestion worker stopped cleanly.")


app = FastAPI(title="Manifold Study Assistant API", lifespan=lifespan)

cors_origins = getattr(
    settings,
    "cors_origins",
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

# Register domain routers
app.include_router(auth.router)
app.include_router(subjects.router)
app.include_router(documents.router)
app.include_router(chats.router)