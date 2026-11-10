# app/routers/documents.py
import asyncio
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from account.auth import get_current_user
from dependencies import (
    ALLOWED_UPLOAD_EXTENSIONS,
    UPLOAD_DIR,
    ingestion_queue,
    logger,
    mongo,
    safe_slug,
    storage,
)
from ingestion.reset_store import delete_by_doc_id
from schemas import IngestionJob

router = APIRouter(tags=["Documents"])


@router.post("/upload", status_code=202)
async def upload_files(
    files: list[UploadFile] = File(...),
    subject: str = Form(...),
    user_id: str = Depends(get_current_user),
):
    subject = subject.strip()
    if not subject:
        raise HTTPException(400, "Subject cannot be empty.")
    if not files:
        raise HTTPException(400, "No files provided.")

    logger.info(
        "[Upload:Received] user_id=%s | subject='%s' | file_count=%d",
        user_id,
        subject,
        len(files),
    )

    await asyncio.to_thread(mongo.ensure_subject, user_id, subject)

    safe_user = safe_slug(user_id)
    safe_subj = safe_slug(subject)
    queued_docs = []

    for upload in files:
        raw_filename = Path(upload.filename or "document.pdf").name
        ext = Path(raw_filename).suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            logger.warning(
                "[Upload:Rejected] Unsupported extension '%s' for file '%s'",
                ext,
                raw_filename,
            )
            raise HTTPException(
                400,
                f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
            )

        guessed_mime, _ = mimetypes.guess_type(raw_filename)
        resolved_content_type = (
            upload.content_type
            if upload.content_type and upload.content_type != "application/octet-stream"
            else (guessed_mime or "application/pdf")
        )

        safe_filename = safe_slug(raw_filename)
        doc_id = f"{safe_user}_{safe_subj}_{safe_filename}"
        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{doc_id}"

        try:
            def _save_to_disk(up=upload, dest=file_path):
                with open(dest, "wb") as buffer:
                    shutil.copyfileobj(up.file, buffer)

            await asyncio.to_thread(_save_to_disk)
            file_size_kb = file_path.stat().st_size / 1024.0

            await asyncio.to_thread(
                mongo.init_document_status,
                doc_id,
                user_id,
                subject,
                raw_filename,
                "queued",
            )
            logger.info(
                "[Upload:Saved] file='%s' (%.1f KB) | mime=%s | doc_id=%s -> status='queued'",
                raw_filename,
                file_size_kb,
                resolved_content_type,
                doc_id,
            )
        except Exception as e:
            file_path.unlink(missing_ok=True)
            logger.exception("[Upload:Error] Failed saving '%s' to disk: %s", raw_filename, e)
            raise HTTPException(
                500, f"Could not save '{raw_filename}' locally: {e}"
            )

        job = IngestionJob(
            file_path=str(file_path),
            clean_name=raw_filename,
            content_type=resolved_content_type,
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
        )
        await ingestion_queue.put(job)

        queued_docs.append(
            {
                "doc_id": doc_id,
                "filename": raw_filename,
                "status": "queued",
            }
        )

    logger.info(
        "[Upload:Queued] Enqueued %d file(s) | current_queue_size=%d",
        len(queued_docs),
        ingestion_queue.qsize(),
    )
    return {
        "user_id": user_id,
        "subject": subject,
        "queue_size": ingestion_queue.qsize(),
        "documents": queued_docs,
    }


@router.get("/status/{doc_id}")
def get_status(doc_id: str, user_id: str = Depends(get_current_user)):
    record = mongo.get_document_status(doc_id)
    if not record or record.get("user_id") != user_id:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")
    return record


@router.get("/documents")
def list_documents(
    subject: Optional[str] = None,
    user_id: str = Depends(get_current_user),
):
    docs = mongo.list_user_documents(user_id, subject)
    logger.debug("[Documents:List] user_id=%s | subject=%s | count=%d", user_id, subject, len(docs))
    return {"documents": docs}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user_id: str = Depends(get_current_user)):
    record = mongo.get_document_status(doc_id)
    if not record or record.get("user_id") != user_id:
        raise HTTPException(404, f"Document '{doc_id}' not found.")

    logger.info("[Documents:Delete] Deleting doc_id=%s for user_id=%s", doc_id, user_id)
    delete_by_doc_id(storage, doc_id)
    mongo.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}