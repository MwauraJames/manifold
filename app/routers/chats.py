# app/routers/chats.py
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from account.auth import get_current_user
from dependencies import _preview, active_streams, logger, mongo
from generation.models import (
    discover_available_models,
    get_default_model,
    resolve_model_config,
)
from pipelines.rag import response_pipeline, stream_response_pipeline
from schemas import NewChatRequest, Question

router = APIRouter(tags=["Chats & Q&A"])


@router.get("/models")
def get_available_models():
    """Returns all LLM models whose provider API keys are currently configured in .env."""
    models = discover_available_models()
    default_model = get_default_model()
    logger.debug("[Models:Discovery] count=%d | default=%s", len(models), default_model)
    return {"models": models, "default": default_model}


@router.post("/chats", status_code=201)
def start_new_chat(
    body: NewChatRequest,
    user_id: str = Depends(get_current_user),
):
    chat_id = str(uuid.uuid4())
    subject = None if body.subject in (None, "", "string") else body.subject.strip()
    doc_id = None if body.doc_id in (None, "", "string") else body.doc_id
    title = None if body.title in (None, "", "string") else body.title

    if subject:
        mongo.ensure_subject(user_id=user_id, subject=subject)

    logger.info(
        "[Chats:New] chat_id=%s | user_id=%s | subject=%s | doc_id=%s",
        chat_id,
        user_id,
        subject,
        doc_id,
    )
    return mongo.ensure_chat_session(
        chat_id=chat_id,
        user_id=user_id,
        subject=subject,
        doc_id=doc_id,
        title=title or "New Study Session",
    )


@router.post("/ask")
def ask(
    body: Question,
    user_id: str = Depends(get_current_user),
):
    try:
        resolved_cfg = resolve_model_config(body.model)
        selected_model = resolved_cfg["model"]
    except ValueError as e:
        logger.warning("[Ask:ModelError] Invalid model '%s': %s", body.model, e)
        raise HTTPException(status_code=400, detail=str(e))

    doc_id = None if body.doc_id in (None, "", "string") else body.doc_id
    subject = None if body.subject in (None, "", "string") else body.subject.strip()
    chat_id = (
        str(uuid.uuid4())
        if body.chat_id in (None, "", "string")
        else body.chat_id
    )

    existing_chat = mongo.get_chat_session(chat_id)
    if existing_chat:
        if existing_chat.get("user_id") != user_id:
            raise HTTPException(403, "Not authorized to access this chat.")
        if not doc_id and existing_chat.get("doc_id"):
            doc_id = existing_chat["doc_id"]
        if not subject and existing_chat.get("subject"):
            subject = existing_chat["subject"]
    else:
        # Only create subject & chat session in MongoDB Atlas if this is a brand-new chat
        if subject:
            mongo.ensure_subject(user_id=user_id, subject=subject)
        mongo.ensure_chat_session(
            chat_id=chat_id,
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
            title=body.query[:60],
        )

    body.doc_id = doc_id
    body.subject = subject

    if body.stream:
        return StreamingResponse(
            stream_response_pipeline(
                body=body,
                user_id=user_id,
                chat_id=chat_id,
                model_name=selected_model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    answer = response_pipeline(
        body=body,
        user_id=user_id,
        chat_id=chat_id,
        model_name=selected_model,
    )
    return {
        "chat_id": chat_id,
        "doc_id": doc_id,
        "subject": subject,
        "model": selected_model,
        "answer": answer,
    }


@router.post("/documents/{doc_id}/ask")
def ask_specific_document(
    doc_id: str,
    body: Question,
    user_id: str = Depends(get_current_user),
):
    body.doc_id = doc_id
    return ask(body, user_id=user_id)


@router.get("/chats")
def get_user_chats(
    subject: Optional[str] = None,
    user_id: str = Depends(get_current_user),
):
    return {"chats": mongo.list_user_chats(user_id, subject)}


@router.get("/chats/{chat_id}/messages")
def get_chat_history(
    chat_id: str,
    user_id: str = Depends(get_current_user),
):
    chat = mongo.get_chat_session(chat_id)
    if not chat or chat.get("user_id") != user_id:
        raise HTTPException(404, f"Chat '{chat_id}' not found.")
    return {
        "chat_id": chat_id,
        "messages": mongo.get_recent_messages(chat_id, limit=100),
    }


@router.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    user_id: str = Depends(get_current_user),
):
    chat = mongo.get_chat_session(chat_id)
    if not chat or chat.get("user_id") != user_id:
        raise HTTPException(404, f"Chat '{chat_id}' not found.")
    logger.info("[Chats:Delete] chat_id=%s | user_id=%s", chat_id, user_id)
    mongo.delete_chat(chat_id)
    return {"status": "deleted", "chat_id": chat_id}


@router.post("/chats/{chat_id}/stop")
def stop_generation(chat_id: str, user_id: str = Depends(get_current_user)):
    chat = mongo.get_chat_session(chat_id)
    if not chat or chat.get("user_id") != user_id:
        raise HTTPException(403, "Not authorized.")

    logger.info("[Chats:Stop] User requested generation stop for chat_id=%s", chat_id)
    active_streams[chat_id] = True
    return {"status": "stopping"}


@router.post("/chats/{chat_id}/edit_last")
def edit_last_prompt(
    chat_id: str,
    body: Question,
    user_id: str = Depends(get_current_user),
):
    existing_chat = mongo.get_chat_session(chat_id)
    if not existing_chat or existing_chat.get("user_id") != user_id:
        raise HTTPException(403, "Not authorized to access this chat.")

    body.doc_id = existing_chat.get("doc_id")
    body.subject = existing_chat.get("subject")

    try:
        resolved_cfg = resolve_model_config(body.model)
        selected_model = resolved_cfg["model"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "[Chats:EditLast] Purging last interaction & re-running | chat_id=%s | new_query='%s'",
        chat_id,
        _preview(body.query, 100),
    )
    mongo.delete_last_interaction(chat_id)

    if body.stream:
        return StreamingResponse(
            stream_response_pipeline(
                body=body,
                user_id=user_id,
                chat_id=chat_id,
                model_name=selected_model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    answer = response_pipeline(body, user_id, chat_id, selected_model)
    return {"chat_id": chat_id, "answer": answer}