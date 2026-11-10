# app/pipelines/rag.py
import json
import time
from typing import Optional

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from dependencies import (
    _preview,
    active_streams,
    co,
    embed_model,
    logger,
    mongo,
    storage,
)
from generation.response import (
    final_response,
    prepare_rolling_memory,
    stream_final_response,
)
from retrieval.expand import route_and_expand_query
from retrieval.hybrid_search import build_hybrid_retriever, merge_leaves_to_parents
from retrieval.reranker import rerank_documents
from schemas import Question


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


def _attach_run_metadata(
    user_id: str, chat_id: str, body: Question, target_model: Optional[str]
):
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

    # 1. Classify intent & rewrite query
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

    # 3. Path B: Specific Fact Lookup (Standard Hybrid + Parent Merge + Cohere Rerank)
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
        text_snip = _preview(
            node.node.get_content() if hasattr(node, "node") else str(node), 90
        )
        logger.info(
            "  -> Leaf[%d] score=%s | text='%s'",
            idx,
            f"{score:.4f}" if score is not None else "N/A",
            text_snip,
        )

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
    logger.info(
        "[Retrieval:Rerank] Sending %d parent chunks to Cohere reranker...",
        len(candidate_texts),
    )
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
    logger.info(
        "[Retrieval:Complete] Final context assembled | chunks=%d | total_chars=%d",
        len(best),
        len(final_ctx),
    )
    return final_ctx


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
        logger.warning(
            "[Ask:Sync] No context found for chat_id=%s; returning fallback response.",
            chat_id,
        )
        answer = "I couldn't find anything relevant to that in your documents."
    else:
        t_gen = time.perf_counter()
        logger.info(
            "[Ask:Generation] Invoking '%s' with context_chars=%d...",
            target_model,
            len(context),
        )
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
    logger.info(
        "[Ask:Saved] Persisted user + assistant messages to MongoDB for chat_id=%s",
        chat_id,
    )

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
        logger.warning(
            "[Ask:Stream] Empty retrieval context for chat_id=%s; streaming fallback message.",
            chat_id,
        )
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