import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import mimetypes
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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

# Keep noisy third-party HTTP loggers at WARNING so your pipeline steps stand out
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _preview(text: Optional[str], max_len: int = 120) -> str:
    """Returns a single-line truncated preview of text for readable terminal logs."""
    if not text:
        return "<empty>"
    cleaned = " ".join(text.strip().split())
    return cleaned if len(cleaned) <= max_len else f"{cleaned[:max_len]}..."


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
from langsmith.run_helpers import get_current_run_tree
from llama_cloud import LlamaCloud
import cohere

from account.auth import supabase, get_current_user
from ingestion.parse import parse_document, is_image_file
from ingestion.chunk import chunk_documents
from ingestion.qdrant_store import Storage
from ingestion.mongo_store import MongoStore
from ingestion.embed import Embed
from ingestion.reset_store import delete_by_doc_id, delete_by_subject, reset_all
from retrieval.reranker import rerank_documents
from generation.response import (
    final_response,
    stream_final_response,
    prepare_rolling_memory,
)
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

active_streams: dict[str, bool] = {}


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
            # 1. Flip status from 'queued' -> 'processing' in MongoDB
            await asyncio.to_thread(mongo.set_document_processing, job.doc_id)
            logger.info("[Worker] MongoDB status set to 'processing' for doc_id=%s", job.doc_id)

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


class ForgotPasswordRequest(BaseModel):
    email: str
    redirect_to: Optional[str] = None


class ResetPasswordVerifyRequest(BaseModel):
    email: str
    token: str
    new_password: str


class UpdatePasswordAuthenticatedRequest(BaseModel):
    access_token: str
    refresh_token: str
    new_password: str


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
logger.info("[Init] Initializing Qdrant Storage, MongoStore, Embed model, LlamaCloud, and Cohere...")
storage = Storage()          # Qdrant (Vectors + Hybrid Search)
mongo = MongoStore()         # MongoDB (Raw PDFs/Images, Parsed Docs, Chat History, Subjects)
embed_model = Embed().get_embedding()
client = LlamaCloud()
co = cohere.ClientV2()
logger.info("[Init] All core clients initialized successfully.")

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


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


@traceable(name="retrieval_pipeline", run_type="chain")
def retrieval_pipeline(
    query: str,
    chat_history: list[dict],
    user_id: str,
    subject: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> str:
    logger.info(
        "[Retrieval:Start] query='%s' | history_msgs=%d | user_id=%s | subject=%s | doc_id=%s",
        _preview(query, 100),
        len(chat_history),
        user_id,
        subject,
        doc_id,
    )

    # 1. Classify intent & rewrite query in a single fast LLM call
    t_route = time.perf_counter()
    route_info = route_and_expand_query(query=query, chat_history=chat_history)
    intent = route_info["intent"]
    expanded_query = route_info["expanded_query"]
    logger.info(
        "[Retrieval:Router] (%.2fs) intent='%s' | expanded_query='%s'",
        time.perf_counter() - t_route,
        intent,
        _preview(expanded_query, 140),
    )

    # Tag the LangSmith span so you can see routing decisions in the trace UI
    rt = get_current_run_tree()
    if rt:
        rt.metadata["query_intent"] = intent
        rt.metadata["expanded_query"] = expanded_query
        rt.tags.append(f"intent:{intent}")

    # 2. Path A: Global Synthesis / Exam Prep / Study Aids
    if intent == "global_synthesis":
        t_syn = time.perf_counter()
        logger.info("[Retrieval:Path A] Fetching stratified synthesis context from MongoDB...")
        synthesis_context = mongo.get_synthesis_context(
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
        )
        if synthesis_context:
            logger.info(
                "[Retrieval:Path A Done] (%.2fs) Returned synthesis context | chars=%d | preview='%s'",
                time.perf_counter() - t_syn,
                len(synthesis_context),
                _preview(synthesis_context, 140),
            )
            return synthesis_context
        logger.warning(
            "[Retrieval:Path A Empty] No ready documents found in MongoDB for global_synthesis; falling back to Path B."
        )

    # 3. Path B: Specific Fact Lookup (Standard Hybrid + Parent Merge + Cohere 0.70)
    t_ret = time.perf_counter()
    logger.info("[Retrieval:Path B] Building Qdrant hybrid retriever...")
    retriever = build_hybrid_retriever(
        storage=storage,
        embed_model=embed_model,
        user_id=user_id,
        subject=subject,
        doc_id=doc_id,
    )
    leaf_results = retriever.retrieve(expanded_query)
    logger.info(
        "[Retrieval:HybridSearch] (%.2fs) Retrieved %d leaf nodes from Qdrant.",
        time.perf_counter() - t_ret,
        len(leaf_results),
    )
    if not leaf_results:
        logger.warning("[Retrieval:Empty] 0 leaf nodes matched in Qdrant.")
        return ""

    for idx, node in enumerate(leaf_results[:5]):
        score = getattr(node, "score", None)
        text_snip = _preview(node.node.get_content() if hasattr(node, "node") else str(node), 90)
        logger.info("  -> Leaf[%d] score=%s | text='%s'", idx, f"{score:.4f}" if score is not None else "N/A", text_snip)

    candidate_texts = merge_leaves_to_parents(leaf_results)
    logger.info(
        "[Retrieval:ParentMerge] Merged %d leaves -> %d unique parent/context chunks.",
        len(leaf_results),
        len(candidate_texts),
    )
    if not candidate_texts:
        logger.warning("[Retrieval:Empty] Parent merge produced 0 candidate chunks.")
        return ""

    t_rerank = time.perf_counter()
    logger.info("[Retrieval:Rerank] Sending %d parent chunks to Cohere reranker...", len(candidate_texts))
    reranked = rerank_documents(co, expanded_query, candidate_texts)
    results_list = reranked.results if hasattr(reranked, "results") else reranked
    logger.info(
        "[Retrieval:Rerank Done] (%.2fs) %d/%d chunks passed reranker threshold.",
        time.perf_counter() - t_rerank,
        len(results_list),
        len(candidate_texts),
    )

    for idx, r in enumerate(results_list):
        rel_score = getattr(r, "relevance_score", None)
        logger.info(
            "  -> Reranked[%d] parent_idx=%d | relevance_score=%s | preview='%s'",
            idx,
            r.index,
            f"{rel_score:.4f}" if rel_score is not None else "N/A",
            _preview(candidate_texts[r.index], 90),
        )

    best = [candidate_texts[r.index] for r in results_list]
    final_ctx = "\n\n---\n\n".join(best)
    logger.info("[Retrieval:Complete] Final context assembled | chunks=%d | total_chars=%d", len(best), len(final_ctx))
    return final_ctx


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
                "session_id": chat_id,
                "subject": body.subject,
                "doc_id": body.doc_id,
                "model": target_model,
            }
        )
        if body.subject:
            rt.tags.append(f"subject:{body.subject}")


@traceable(
    name="Manifold",
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
    logger.info(
        "[Ask:Sync] chat_id=%s | model=%s | subject=%s | doc_id=%s | query='%s'",
        chat_id,
        target_model,
        body.subject,
        body.doc_id,
        _preview(body.query, 100),
    )
    _attach_run_metadata(user_id, chat_id, body, target_model)

    summary, recent_messages = prepare_rolling_memory(
        mongo=mongo,
        chat_id=chat_id,
    )
    logger.info(
        "[Ask:Memory] chat_id=%s | has_summary=%s (chars=%d) | recent_messages=%d",
        chat_id,
        bool(summary),
        len(summary) if summary else 0,
        len(recent_messages),
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
        logger.warning("[Ask:Sync] No context found for chat_id=%s; returning fallback response.", chat_id)
        answer = "I couldn't find anything relevant to that in your documents."
    else:
        t_gen = time.perf_counter()
        logger.info("[Ask:Generation] Invoking '%s' with context_chars=%d...", target_model, len(context))
        answer = final_response(
            model_name=target_model,
            query=body.query,
            context=context,
            chat_history=recent_messages,
            summary=summary,
        )
        logger.info(
            "[Ask:Generation Done] (%.2fs) answer_chars=%d | preview='%s'",
            time.perf_counter() - t_gen,
            len(answer),
            _preview(answer, 120),
        )

    mongo.add_message(chat_id, user_id, "user", body.query)
    mongo.add_message(chat_id, user_id, "assistant", answer)
    logger.info("[Ask:Saved] Persisted user + assistant messages to MongoDB for chat_id=%s", chat_id)

    return {"messages": [{"role": "assistant", "content": answer}]}


@traceable(
    name="Manifold",
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
    logger.info(
        "[Ask:Stream] START | chat_id=%s | model=%s | subject=%s | doc_id=%s | query='%s'",
        chat_id,
        target_model,
        body.subject,
        body.doc_id,
        _preview(body.query, 100),
    )
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
    logger.info(
        "[Ask:Memory] chat_id=%s | has_summary=%s (chars=%d) | recent_messages=%d",
        chat_id,
        bool(summary),
        len(summary) if summary else 0,
        len(recent_messages),
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
    active_streams[chat_id] = False

    if not context:
        logger.warning("[Ask:Stream] Empty retrieval context for chat_id=%s; streaming fallback message.", chat_id)
        fallback = "I couldn't find anything relevant to that in your documents."
        full_answer_parts.append(fallback)
        chunk_payload = json.dumps({"type": "token", "content": fallback})
        yield f"data: {chunk_payload}\n\n"
    else:
        t_stream = time.perf_counter()
        logger.info(
            "[Ask:Stream] Streaming tokens from '%s' | context_chars=%d",
            target_model,
            len(context),
        )
        token_chunks = 0
        for token in stream_final_response(
            model_name=target_model,
            query=body.query,
            context=context,
            chat_history=recent_messages,
            summary=summary,
        ):
            if active_streams.get(chat_id):
                logger.info("[Ask:Stream] Stop signal detected for chat_id=%s", chat_id)
                stop_msg = "\n\n*[Generation stopped by user]*"
                full_answer_parts.append(stop_msg)
                yield f"data: {json.dumps({'type': 'token', 'content': stop_msg})}\n\n"
                break

            token_chunks += 1
            full_answer_parts.append(token)
            chunk_payload = json.dumps({"type": "token", "content": token})
            yield f"data: {chunk_payload}\n\n"

        logger.info(
            "[Ask:Stream] Finished LLM stream in %.2fs | chunks=%d",
            time.perf_counter() - t_stream,
            token_chunks,
        )

    active_streams.pop(chat_id, None)

    full_answer = "".join(full_answer_parts)
    mongo.add_message(chat_id, user_id, "user", body.query)
    mongo.add_message(chat_id, user_id, "assistant", full_answer)
    logger.info(
        "[Ask:Stream] COMPLETE | chat_id=%s | answer_chars=%d | preview='%s'",
        chat_id,
        len(full_answer),
        _preview(full_answer, 120),
    )

    yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id})}\n\n"


# =====================================================================
# AUTHENTICATION ENDPOINTS (Supabase Email + Password)
# =====================================================================

@app.post("/auth/signup", status_code=201)
def sign_up(credentials: AuthCredentials):
    logger.info("[Auth:Signup] Attempting signup for email=%s", credentials.email)
    try:
        res = supabase.auth.sign_up(
            {"email": credentials.email, "password": credentials.password}
        )
        if not res.user:
            raise HTTPException(400, "Could not register user.")

        logger.info("[Auth:Signup] Registered user_id=%s | email=%s", res.user.id, res.user.email)
        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "access_token": res.session.access_token if res.session else None,
            "refresh_token": res.session.refresh_token if res.session else None,
            "confirmation_required": res.session is None,
        }
    except Exception as e:
        logger.warning("[Auth:Signup] Failed for email=%s: %s", credentials.email, e)
        raise HTTPException(400, f"Sign-up failed: {str(e)}")


@app.post("/auth/login")
def login(credentials: AuthCredentials):
    logger.info("[Auth:Login] Attempting login for email=%s", credentials.email)
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": credentials.email, "password": credentials.password}
        )
        logger.info("[Auth:Login] Success for user_id=%s", res.user.id)
        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "expires_in": res.session.expires_in,
        }
    except Exception as e:
        logger.warning("[Auth:Login] Failed for email=%s: %s", credentials.email, e)
        raise HTTPException(401, f"Invalid email or password: {str(e)}")


@app.post("/auth/refresh")
def refresh_session(body: RefreshRequest):
    try:
        res = supabase.auth.refresh_session(body.refresh_token)
        logger.info("[Auth:Refresh] Token refreshed for user_id=%s", res.user.id)
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user_id": res.user.id,
        }
    except Exception as e:
        logger.warning("[Auth:Refresh] Failed: %s", e)
        raise HTTPException(401, f"Could not refresh session: {str(e)}")


@app.get("/auth/me")
def get_me(user_id: str = Depends(get_current_user)):
    return {"user_id": user_id}


@app.post("/auth/forgot-password")
def forgot_password(body: ForgotPasswordRequest):
    logger.info("[Auth:ForgotPassword] Requested for email=%s", body.email)
    try:
        options = {"redirect_to": body.redirect_to} if body.redirect_to else {}
        supabase.auth.reset_password_for_email(body.email.strip(), options=options)
        return {
            "status": "sent",
            "message": "If an account with that email exists, a password recovery code/link has been sent.",
        }
    except Exception as e:
        logger.warning("[Auth:ForgotPassword] Error for email=%s: %s", body.email, e)
        raise HTTPException(400, f"Could not initiate password reset: {str(e)}")


@app.post("/auth/reset-password")
def reset_password_with_otp(body: ResetPasswordVerifyRequest):
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters.")

    try:
        verify_res = supabase.auth.verify_otp(
            {
                "email": body.email.strip(),
                "token": body.token.strip(),
                "type": "recovery",
            }
        )
        if not verify_res.session:
            raise HTTPException(400, "Invalid or expired recovery code.")

        supabase.auth.set_session(
            verify_res.session.access_token,
            verify_res.session.refresh_token,
        )
        supabase.auth.update_user({"password": body.new_password})
        logger.info("[Auth:ResetPassword] Password updated via OTP for user_id=%s", verify_res.user.id)

        return {
            "status": "password_updated",
            "user_id": verify_res.user.id,
            "email": verify_res.user.email,
            "access_token": verify_res.session.access_token,
            "refresh_token": verify_res.session.refresh_token,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[Auth:ResetPassword] Failed for email=%s: %s", body.email, e)
        raise HTTPException(400, f"Password reset failed: {str(e)}")


@app.post("/auth/update-password")
def update_password_with_tokens(body: UpdatePasswordAuthenticatedRequest):
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters.")

    try:
        supabase.auth.set_session(body.access_token, body.refresh_token)
        res = supabase.auth.update_user({"password": body.new_password})
        if not res.user:
            raise HTTPException(400, "Could not update password.")

        logger.info("[Auth:UpdatePassword] Password updated via token for user_id=%s", res.user.id)
        return {
            "status": "password_updated",
            "user_id": res.user.id,
            "email": res.user.email,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[Auth:UpdatePassword] Failed: %s", e)
        raise HTTPException(401, f"Invalid or expired recovery session: {str(e)}")


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

    logger.info("[Subjects:Create] user_id=%s | subject='%s'", user_id, subject_name)
    return mongo.ensure_subject(
        user_id=user_id,
        subject=subject_name,
        description=body.description,
    )


@app.get("/subjects")
def list_subjects(user_id: str = Depends(get_current_user)):
    subjects = mongo.list_user_subjects(user_id)
    logger.debug("[Subjects:List] user_id=%s | count=%d", user_id, len(subjects))
    return {"subjects": subjects}


@app.delete("/subjects/{subject:path}")
def delete_subject(
    subject: str,
    user_id: str = Depends(get_current_user),
):
    logger.info("[Subjects:Delete] Deleting subject='%s' for user_id=%s", subject, user_id)
    mongo_stats = mongo.delete_subject(user_id=user_id, subject=subject)
    delete_by_subject(
        storage=storage,
        user_id=user_id,
        subject=subject,
        doc_ids=mongo_stats["doc_ids"],
    )
    logger.info("[Subjects:Delete Done] subject='%s' | stats=%s", subject, mongo_stats)
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
    docs = mongo.list_user_documents(user_id, subject)
    logger.debug("[Documents:List] user_id=%s | subject=%s | count=%d", user_id, subject, len(docs))
    return {"documents": docs}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, user_id: str = Depends(get_current_user)):
    record = mongo.get_document_status(doc_id)
    if not record or record.get("user_id") != user_id:
        raise HTTPException(404, f"Document '{doc_id}' not found.")

    logger.info("[Documents:Delete] Deleting doc_id=%s for user_id=%s", doc_id, user_id)
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

    logger.info("[Chats:New] chat_id=%s | user_id=%s | subject=%s | doc_id=%s", chat_id, user_id, subject, doc_id)
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
    default_model = get_default_model()
    logger.debug("[Models:Discovery] count=%d | default=%s", len(models), default_model)
    return {"models": models, "default": default_model}


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
    logger.info("[Chats:Delete] chat_id=%s | user_id=%s", chat_id, user_id)
    mongo.delete_chat(chat_id)
    return {"status": "deleted", "chat_id": chat_id}


@app.post("/chats/{chat_id}/stop")
def stop_generation(chat_id: str, user_id: str = Depends(get_current_user)):
    chat = mongo.get_chat_session(chat_id)
    if not chat or chat.get("user_id") != user_id:
        raise HTTPException(403, "Not authorized.")

    logger.info("[Chats:Stop] User requested generation stop for chat_id=%s", chat_id)
    active_streams[chat_id] = True
    return {"status": "stopping"}


@app.post("/chats/{chat_id}/edit_last")
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