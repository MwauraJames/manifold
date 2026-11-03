import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_index.core import Document
from langsmith import traceable
from llama_cloud import LlamaCloud
import cohere

from account.auth import supabase, get_current_user
from ingestion.parse import parse_document
from ingestion.chunk import chunk_documents
from ingestion.storage import Storage
from ingestion.mongo_store import MongoStore
from ingestion.embed import Embed
from ingestion.reset_store import delete_by_doc_id, delete_by_subject, reset_all
from retrieval.reranker import rerank_documents
from generation.response import (
    final_response,
    stream_final_response,
    prepare_rolling_memory,
)
# In app/main.py imports:
from retrieval.expand import expand_query, route_and_expand_query
from retrieval.hybrid_search import build_hybrid_retriever, merge_leaves_to_parents
from generation.models import (
    discover_available_models,
    get_default_model,
    resolve_model_config,
)
from configs.config import Settings


# ---------- Sequential Ingestion Queue ----------
@dataclass
class IngestionJob:
    file_path: str
    clean_name: str
    content_type: str
    user_id: str
    subject: str
    doc_id: str


ingestion_queue: asyncio.Queue[IngestionJob] = asyncio.Queue()


async def ingestion_worker():
    """
    Single background worker that processes uploaded documents strictly
    ONE AFTER ANOTHER (FIFO) so CPU/RAM and external APIs are never overwhelmed.
    """
    while True:
        job: IngestionJob = await ingestion_queue.get()
        try:
            # 1. Flip status from 'queued' -> 'processing' in MongoDB
            await asyncio.to_thread(mongo.set_document_processing, job.doc_id)

            # 2. Run the synchronous ingest_pipeline in a worker thread
            await asyncio.to_thread(
                ingest_pipeline,
                job.file_path,
                job.clean_name,
                job.content_type,
                job.user_id,
                job.subject,
                job.doc_id,
            )
        except Exception as e:
            print(f"[IngestionWorker] Error processing {job.doc_id}: {e}")
        finally:
            ingestion_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the sequential ingestion worker when FastAPI boots up."""
    worker_task = asyncio.create_task(ingestion_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)

settings = Settings()

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


# ---------- Request Schemas ----------
class AuthCredentials(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class SubjectCreate(BaseModel):
    subject: str
    description: Optional[str] = None


class Question(BaseModel):
    query: str
    subject: Optional[str] = None
    doc_id: Optional[str] = None
    chat_id: Optional[str] = None
    model: Optional[str] = None
    stream: bool = True


class NewChatRequest(BaseModel):
    subject: Optional[str] = None
    doc_id: Optional[str] = None
    title: Optional[str] = None


# ---------- Clients & Directories ----------
storage = Storage()          # Qdrant (Vectors + Hybrid Search)
mongo = MongoStore()         # MongoDB (Raw PDFs, Parsed Docs, Chat History, Subjects)
embed_model = Embed().get_embedding()
client = LlamaCloud()
co = cohere.ClientV2()

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def safe_slug(value: str) -> str:
    """
    Replaces slashes, backslashes, spaces, and unsafe path characters with '_'
    so subject names like 'CS 101/AI' never break file paths or URLs.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


# ---------- Pipelines ----------
@traceable(name="ingest_pipeline", run_type="chain")
def ingest_pipeline(
    file_path: str,
    clean_name: str,
    content_type: str,
    user_id: str,
    subject: str,
    doc_id: str,
):
    try:
        parsed_content = parse_document(client, file_path)

        delete_by_doc_id(storage, doc_id)
        stats = chunk_documents(
            storage=storage,
            embed_model=embed_model,
            docs=[Document(text=parsed_content)],
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
        )

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
    except Exception as e:
        mongo.mark_document_failed(doc_id, str(e))
        try:
            delete_by_doc_id(storage, doc_id)
        except Exception:
            pass
        raise
    finally:
        Path(file_path).unlink(missing_ok=True)


@traceable(name="retrieval_pipeline", run_type="chain")
def retrieval_pipeline(
    query: str,
    chat_history: list[dict],
    user_id: str,
    subject: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> str:
    # 1. Classify intent & rewrite query in a single fast LLM call
    route_info = route_and_expand_query(query=query, chat_history=chat_history)
    intent = route_info["intent"]
    expanded_query = route_info["expanded_query"]

    # Tag the LangSmith span so you can see routing decisions in the trace UI
    rt = get_current_run_tree()
    if rt:
        rt.metadata["query_intent"] = intent
        rt.metadata["expanded_query"] = expanded_query
        rt.tags.append(f"intent:{intent}")

    # 2. Path A: Global Synthesis / Exam Prep / Study Aids
    # Bypasses Qdrant similarity & Cohere 0.70 threshold, sampling directly from MongoDB
    if intent == "global_synthesis":
        synthesis_context = mongo.get_synthesis_context(
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
        )
        if synthesis_context:
            return synthesis_context

    # 3. Path B: Specific Fact Lookup (Standard Hybrid + Parent Merge + Cohere 0.70)
    retriever = build_hybrid_retriever(
        storage=storage,
        embed_model=embed_model,
        user_id=user_id,
        subject=subject,
        doc_id=doc_id,
    )
    leaf_results = retriever.retrieve(expanded_query)
    if not leaf_results:
        return ""

    candidate_texts = merge_leaves_to_parents(leaf_results)
    if not candidate_texts:
        return ""

    reranked = rerank_documents(co, expanded_query, candidate_texts)
    results_list = reranked.results if hasattr(reranked, "results") else reranked
    best = [candidate_texts[r.index] for r in results_list]

    return "\n\n---\n\n".join(best)


from langsmith.run_helpers import get_current_run_tree


def _format_root_inputs(inputs: dict) -> dict:
    """Formats the root run inputs as a clean User chat bubble in LangSmith."""
    body = inputs.get("body")
    query_text = body.query if hasattr(body, "query") else str(body)
    return {"messages": [{"role": "user", "content": query_text}]}


def _reduce_sse_stream_for_langsmith(sse_chunks: list[str]) -> dict:
    """Combines yielded SSE 'data: ...' lines into a single AI chat bubble in LangSmith."""
    tokens: list[str] = []
    for raw_line in sse_chunks:
        if raw_line.startswith("data: "):
            try:
                payload = json.loads(raw_line[len("data: ") :].strip())
                if payload.get("type") == "token":
                    tokens.append(payload.get("content", ""))
            except Exception:
                pass
    return {"messages": [{"role": "assistant", "content": "".join(tokens)}]}


def _attach_run_metadata(user_id: str, chat_id: str, body: Question, target_model: Optional[str]):
    """Attaches user_id, chat_id, subject, and session_id to LangSmith metadata & tags."""
    rt = get_current_run_tree()
    if rt:
        rt.metadata.update(
            {
                "user_id": user_id,
                "chat_id": chat_id,
                "session_id": chat_id,  # Groups multi-turn chats in LangSmith Threads tab
                "subject": body.subject,
                "doc_id": body.doc_id,
                "model": target_model,
            }
        )
        if body.subject:
            rt.tags.append(f"subject:{body.subject}")


@traceable(
    name="StudyAssistant",
    run_type="chain",
    process_inputs=_format_root_inputs,
)
def response_pipeline(
    body: Question,
    user_id: str,
    chat_id: str,
    model_name: Optional[str] = None,
) -> dict:
    target_model = model_name or body.model
    _attach_run_metadata(user_id, chat_id, body, target_model)

    summary, recent_messages = prepare_rolling_memory(
        mongo=mongo,
        chat_id=chat_id,
    )

    history_for_expansion = []
    if summary:
        history_for_expansion.append(
            {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
        )
    history_for_expansion.extend(recent_messages)

    context = retrieval_pipeline(
        query=body.query,
        chat_history=history_for_expansion,
        user_id=user_id,
        subject=body.subject,
        doc_id=body.doc_id,
    )

    if not context:
        answer = "I couldn't find anything relevant to that in your documents."
    else:
        answer = final_response(
            model_name=target_model,
            query=body.query,
            context=context,
            chat_history=recent_messages,
            summary=summary,
        )

    mongo.add_message(chat_id, user_id, "user", body.query)
    mongo.add_message(chat_id, user_id, "assistant", answer)

    return {"messages": [{"role": "assistant", "content": answer}]}


@traceable(
    name="StudyAssistant",
    run_type="chain",
    process_inputs=_format_root_inputs,
    reduce_fn=_reduce_sse_stream_for_langsmith,
)
def stream_response_pipeline(
    body: Question,
    user_id: str,
    chat_id: str,
    model_name: Optional[str] = None,
):
    target_model = model_name or body.model
    _attach_run_metadata(user_id, chat_id, body, target_model)

    meta_payload = json.dumps(
        {
            "type": "meta",
            "chat_id": chat_id,
            "doc_id": body.doc_id,
            "subject": body.subject,
            "model": target_model,
        }
    )
    yield f"data: {meta_payload}\n\n"

    summary, recent_messages = prepare_rolling_memory(
        mongo=mongo,
        chat_id=chat_id,
    )

    history_for_expansion = []
    if summary:
        history_for_expansion.append(
            {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
        )
    history_for_expansion.extend(recent_messages)

    context = retrieval_pipeline(
        query=body.query,
        chat_history=history_for_expansion,
        user_id=user_id,
        subject=body.subject,
        doc_id=body.doc_id,
    )

    full_answer_parts: list[str] = []

    if not context:
        fallback = "I couldn't find anything relevant to that in your documents."
        full_answer_parts.append(fallback)
        chunk_payload = json.dumps({"type": "token", "content": fallback})
        yield f"data: {chunk_payload}\n\n"
    else:
        for token in stream_final_response(
            model_name=target_model,
            query=body.query,
            context=context,
            chat_history=recent_messages,
            summary=summary,
        ):
            full_answer_parts.append(token)
            chunk_payload = json.dumps({"type": "token", "content": token})
            yield f"data: {chunk_payload}\n\n"

    full_answer = "".join(full_answer_parts)
    mongo.add_message(chat_id, user_id, "user", body.query)
    mongo.add_message(chat_id, user_id, "assistant", full_answer)

    yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id})}\n\n"

# =====================================================================
# AUTHENTICATION ENDPOINTS (Supabase Email + Password)
# =====================================================================

@app.post("/auth/signup", status_code=201)
def sign_up(credentials: AuthCredentials):
    try:
        res = supabase.auth.sign_up(
            {"email": credentials.email, "password": credentials.password}
        )
        if not res.user:
            raise HTTPException(400, "Could not register user.")

        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "access_token": res.session.access_token if res.session else None,
            "refresh_token": res.session.refresh_token if res.session else None,
            "confirmation_required": res.session is None,
        }
    except Exception as e:
        raise HTTPException(400, f"Sign-up failed: {str(e)}")


@app.post("/auth/login")
def login(credentials: AuthCredentials):
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": credentials.email, "password": credentials.password}
        )
        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "expires_in": res.session.expires_in,
        }
    except Exception as e:
        raise HTTPException(401, f"Invalid email or password: {str(e)}")


@app.post("/auth/refresh")
def refresh_session(body: RefreshRequest):
    try:
        res = supabase.auth.refresh_session(body.refresh_token)
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user_id": res.user.id,
        }
    except Exception as e:
        raise HTTPException(401, f"Could not refresh session: {str(e)}")


@app.get("/auth/me")
def get_me(user_id: str = Depends(get_current_user)):
    return {"user_id": user_id}


# =====================================================================
# PROTECTED SUBJECT ENDPOINTS
# =====================================================================

@app.post("/subjects", status_code=201)
def create_subject(
    body: SubjectCreate,
    user_id: str = Depends(get_current_user),
):
    subject_name = body.subject.strip()
    if not subject_name or subject_name == "string":
        raise HTTPException(400, "Subject name cannot be empty.")

    return mongo.ensure_subject(
        user_id=user_id,
        subject=subject_name,
        description=body.description,
    )


@app.get("/subjects")
def list_subjects(user_id: str = Depends(get_current_user)):
    return {"subjects": mongo.list_user_subjects(user_id)}


@app.delete("/subjects/{subject:path}")
def delete_subject(
    subject: str,
    user_id: str = Depends(get_current_user),
):
    mongo_stats = mongo.delete_subject(user_id=user_id, subject=subject)
    delete_by_subject(
        storage=storage,
        user_id=user_id,
        subject=subject,
        doc_ids=mongo_stats["doc_ids"],
    )
    return {
        "status": "deleted",
        "user_id": user_id,
        "subject": subject,
        **mongo_stats,
    }


# =====================================================================
# PROTECTED DOCUMENT ENDPOINTS
# =====================================================================

@app.post("/upload", status_code=202)
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

    await asyncio.to_thread(mongo.ensure_subject, user_id, subject)

    safe_user = safe_slug(user_id)
    safe_subj = safe_slug(subject)
    queued_docs = []

    for upload in files:
        raw_filename = Path(upload.filename or "document.pdf").name
        safe_filename = safe_slug(raw_filename)
        doc_id = f"{safe_user}_{safe_subj}_{safe_filename}"
        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{doc_id}"

        try:
            def _save_to_disk(up=upload, dest=file_path):
                with open(dest, "wb") as buffer:
                    shutil.copyfileobj(up.file, buffer)

            await asyncio.to_thread(_save_to_disk)

            await asyncio.to_thread(
                mongo.init_document_status,
                doc_id,
                user_id,
                subject,
                raw_filename,
                "queued",
            )
        except Exception as e:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                500, f"Could not save '{raw_filename}' locally: {e}"
            )

        job = IngestionJob(
            file_path=str(file_path),
            clean_name=raw_filename,
            content_type=upload.content_type or "application/pdf",
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

    return {
        "user_id": user_id,
        "subject": subject,
        "queue_size": ingestion_queue.qsize(),
        "documents": queued_docs,
    }


@app.get("/status/{doc_id}")
def get_status(doc_id: str, user_id: str = Depends(get_current_user)):
    record = mongo.get_document_status(doc_id)
    if not record or record.get("user_id") != user_id:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")
    return record


@app.get("/documents")
def list_documents(
    subject: Optional[str] = None,
    user_id: str = Depends(get_current_user),
):
    return {"documents": mongo.list_user_documents(user_id, subject)}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user_id: str = Depends(get_current_user)):
    record = mongo.get_document_status(doc_id)
    if not record or record.get("user_id") != user_id:
        raise HTTPException(404, f"Document '{doc_id}' not found.")

    delete_by_doc_id(storage, doc_id)
    mongo.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# =====================================================================
# PROTECTED CHAT ENDPOINTS
# =====================================================================

@app.post("/chats", status_code=201)
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

    return mongo.ensure_chat_session(
        chat_id=chat_id,
        user_id=user_id,
        subject=subject,
        doc_id=doc_id,
        title=title or "New Study Session",
    )


# =====================================================================
# MODEL DISCOVERY ENDPOINT
# =====================================================================

@app.get("/models")
def get_available_models():
    """Returns all LLM models whose provider API keys are currently configured in .env."""
    models = discover_available_models()
    return {"models": models, "default": get_default_model()}


# =====================================================================
# UPDATED /ask ENDPOINT (Supports Model Selection + SSE Streaming)
# =====================================================================

@app.post("/ask")
def ask(
    body: Question,
    user_id: str = Depends(get_current_user),
):
    try:
        resolved_cfg = resolve_model_config(body.model)
        selected_model = resolved_cfg["model"]
    except ValueError as e:
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

    body.doc_id = doc_id
    body.subject = subject

    if subject:
        mongo.ensure_subject(user_id=user_id, subject=subject)

    mongo.ensure_chat_session(
        chat_id=chat_id,
        user_id=user_id,
        subject=subject,
        doc_id=doc_id,
        title=body.query[:60],
    )

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


@app.post("/documents/{doc_id}/ask")
def ask_specific_document(
    doc_id: str,
    body: Question,
    user_id: str = Depends(get_current_user),
):
    body.doc_id = doc_id
    return ask(body, user_id=user_id)


@app.get("/chats")
def get_user_chats(
    subject: Optional[str] = None,
    user_id: str = Depends(get_current_user),
):
    return {"chats": mongo.list_user_chats(user_id, subject)}


@app.get("/chats/{chat_id}/messages")
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


@app.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    user_id: str = Depends(get_current_user),
):
    chat = mongo.get_chat_session(chat_id)
    if not chat or chat.get("user_id") != user_id:
        raise HTTPException(404, f"Chat '{chat_id}' not found.")
    mongo.delete_chat(chat_id)
    return {"status": "deleted", "chat_id": chat_id}