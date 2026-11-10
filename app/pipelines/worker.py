# app/pipelines/worker.py
import asyncio
import time
from pathlib import Path

from langsmith import traceable
from llama_index.core import Document

from dependencies import (
    _preview,
    client,
    embed_model,
    ingestion_queue,
    logger,
    mongo,
    storage,
)
from ingestion.chunk import chunk_documents
from ingestion.parse import is_image_file, parse_document
from ingestion.reset_store import delete_by_doc_id
from schemas import IngestionJob


@traceable(name="ingest_pipeline", run_type="chain")
def ingest_pipeline(
    file_path: str,
    clean_name: str,
    content_type: str,
    user_id: str,
    subject: str,
    doc_id: str,
):
    is_img = is_image_file(file_path, content_type=content_type)
    logger.info(
        "[Ingest:Start] doc_id=%s | file='%s' | content_type=%s | route=%s | user_id=%s | subject='%s'",
        doc_id,
        clean_name,
        content_type,
        "Vision-LLM (Image)" if is_img else "LlamaParse (Document)",
        user_id,
        subject,
    )

    try:
        # Step 1: Parse Document or Image -> Markdown
        t_parse = time.perf_counter()
        logger.info("[Ingest:Step 1/4] Parsing '%s' into Markdown...", clean_name)
        parsed_content = parse_document(
            client,
            file_path,
            content_type=content_type,
        )
        parse_elapsed = time.perf_counter() - t_parse
        logger.info(
            "[Ingest:Step 1/4 Done] Parsed '%s' in %.2fs | chars=%d | preview='%s'",
            clean_name,
            parse_elapsed,
            len(parsed_content),
            _preview(parsed_content, 150),
        )

        # Step 2: Clear any existing Qdrant vectors for this doc_id
        logger.info("[Ingest:Step 2/4] Clearing stale Qdrant vectors for doc_id=%s", doc_id)
        delete_by_doc_id(storage, doc_id)

        # Step 3: Hierarchical Chunking & Hybrid Embedding into Qdrant
        t_chunk = time.perf_counter()
        logger.info("[Ingest:Step 3/4] Chunking & embedding nodes into Qdrant for doc_id=%s...", doc_id)
        stats = chunk_documents(
            storage=storage,
            embed_model=embed_model,
            docs=[Document(text=parsed_content)],
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
        )
        chunk_elapsed = time.perf_counter() - t_chunk
        logger.info(
            "[Ingest:Step 3/4 Done] Qdrant indexing finished in %.2fs | stats=%s",
            chunk_elapsed,
            stats,
        )

        # Step 4: Finalize in MongoDB (GridFS raw binary + parsed_markdown)
        t_mongo = time.perf_counter()
        logger.info("[Ingest:Step 4/4] Streaming raw file to GridFS & saving markdown in MongoDB...")
        mongo.finalize_document(
            doc_id=doc_id,
            user_id=user_id,
            subject=subject,
            filename=clean_name,
            file_path=file_path,
            content_type=content_type,
            parsed_markdown=parsed_content,
            num_leaves=stats["leaves"],
        )
        mongo_elapsed = time.perf_counter() - t_mongo
        logger.info(
            "[Ingest:Complete] doc_id=%s marked 'ready' in MongoDB (%.2fs) | num_leaves=%d",
            doc_id,
            mongo_elapsed,
            stats["leaves"],
        )
    except Exception as e:
        logger.exception("[Ingest:Error] Pipeline failed for doc_id=%s: %s", doc_id, e)
        mongo.mark_document_failed(doc_id, str(e))
        try:
            delete_by_doc_id(storage, doc_id)
            logger.info("[Ingest:Rollback] Cleaned up partial Qdrant points for doc_id=%s", doc_id)
        except Exception as cleanup_err:
            logger.warning(
                "[Ingest:Rollback] Failed cleaning Qdrant points for doc_id=%s: %s",
                doc_id,
                cleanup_err,
            )
        raise
    finally:
        Path(file_path).unlink(missing_ok=True)
        logger.debug("[Ingest:Cleanup] Removed temporary file %s", file_path)


async def ingestion_worker():
    """
    Single background worker that processes uploaded documents strictly
    ONE AFTER ANOTHER (FIFO) so CPU/RAM and external APIs are never overwhelmed.
    """
    logger.info("[Worker] Background FIFO ingestion worker started and waiting for jobs.")
    while True:
        job: IngestionJob = await ingestion_queue.get()
        logger.info(
            "[Worker] Popped job from queue | doc_id=%s | file=%s | mime=%s | remaining_queue=%d",
            job.doc_id,
            job.clean_name,
            job.content_type,
            ingestion_queue.qsize(),
        )
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(mongo.set_document_processing, job.doc_id)
            logger.info("[Worker] MongoDB status set to 'processing' for doc_id=%s", job.doc_id)

            await asyncio.to_thread(
                ingest_pipeline,
                job.file_path,
                job.clean_name,
                job.content_type,
                job.user_id,
                job.subject,
                job.doc_id,
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "[Worker] Successfully completed job | doc_id=%s | total_time=%.2fs",
                job.doc_id,
                elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception(
                "[Worker] FAILED processing job | doc_id=%s | elapsed=%.2fs | error=%s",
                job.doc_id,
                elapsed,
                e,
            )
        finally:
            ingestion_queue.task_done()