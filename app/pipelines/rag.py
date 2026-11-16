# app/pipelines/rag.py
import json
import re
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
from ingestion.chunk import _extract_section_header
from retrieval.expand import route_and_expand_query
from retrieval.hybrid_search import build_hybrid_retriever, merge_leaves_to_parents
from retrieval.reranker import rerank_documents
from schemas import Question

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def _format_root_inputs(inputs: dict) -> dict:
    """Formats the root run inputs as a clean User chat bubble in LangSmith."""
    body = inputs.get("body")
    query_text = body.query if hasattr(body, "query") else str(body)
    return {"messages": [{"role": "user", "content": query_text}]}


def _reduce_sse_stream_for_langsmith(sse_chunks: list[str]) -> dict:
    """Combines yielded SSE 'data: ...' lines into a single AI chat bubble in LangSmith."""
    tokens: list[str] = []
    citations: list[dict] = []
    for raw_line in sse_chunks:
        if raw_line.startswith("data: "):
            try:
                payload = json.loads(raw_line[len("data: ") :].strip())
                if payload.get("type") == "token":
                    tokens.append(payload.get("content", ""))
                elif payload.get("type") == "done" and payload.get("citations"):
                    citations = payload["citations"]
            except Exception:
                pass
    return {
        "messages": [{"role": "assistant", "content": "".join(tokens)}],
        "citations": citations,
    }


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


def _build_numbered_context(
    raw_sources: list[dict],
    retrieval_mode: str,
) -> tuple[str, list[dict]]:
    """
    Wraps each retrieved chunk in a numbered <source id="N"> XML block for the LLM
    and constructs the clean citation metadata list for the UI and MongoDB.
    """
    if not raw_sources:
        return "", []

    # Backfill filenames from MongoDB for any legacy Qdrant chunks missing 'filename'
    missing_doc_ids = [
        s["doc_id"]
        for s in raw_sources
        if s.get("doc_id") and (not s.get("filename") or s.get("filename") == s.get("doc_id"))
    ]
    filename_map = mongo.get_filenames_by_doc_ids(missing_doc_ids) if missing_doc_ids else {}

    context_blocks: list[str] = []
    citations: list[dict] = []

    for idx, src in enumerate(raw_sources, start=1):
        text = (src.get("text") or "").strip()
        doc_id = src.get("doc_id")
        filename = (
            src.get("filename")
            if src.get("filename") and src.get("filename") != doc_id
            else filename_map.get(doc_id) or doc_id or f"Source {idx}"
        )
        subject = src.get("subject")
        section = (
            src.get("section")
            or _extract_section_header(text)
            or ("Full-Document Sample" if retrieval_mode == "global_synthesis" else f"Passage {idx}")
        )
        score = src.get("rerank_score")
        if score is None:
            score = src.get("leaf_score")

        context_blocks.append(
            f'<source id="{idx}" file="{filename}" section="{section}">\n'
            f"{text}\n"
            f"</source>"
        )

        citations.append({
            "id": idx,
            "doc_id": doc_id,
            "filename": filename,
            "subject": subject,
            "section": section,
            "score": round(float(score), 4) if score is not None else None,
            "retrieval_mode": retrieval_mode,
            "snippet": text[:600] + ("..." if len(text) > 600 else ""),
            "cited": False,
        })

    return "\n\n".join(context_blocks), citations


def _mark_cited_sources(answer: str, citations: list[dict]) -> list[dict]:
    """Marks which retrieved sources were explicitly cited as [N] in the generated answer."""
    if not citations:
        return []
    cited_ids = {int(m) for m in _CITATION_MARKER_RE.findall(answer or "")}
    updated = []
    for c in citations:
        item = dict(c)
        item["cited"] = item["id"] in cited_ids
        updated.append(item)
    return updated


@traceable(name="retrieval_pipeline", run_type="chain")
def retrieval_pipeline(
    query: str,
    chat_history: list[dict],
    user_id: str,
    subject: Optional[str] = None,
    doc_id: Optional[str] = None,
):
    """
    Yields ('status', message) during execution to update the UI step-by-step,
    and finishes by yielding ('result', (final_ctx, citations)).
    """
    yield ("status", "Analyzing query intent…")
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

    if intent == "global_synthesis":
        yield ("status", "Sampling full-document sections for global synthesis…")
        t_syn = time.perf_counter()
        synthesis_sources = mongo.get_synthesis_sources(
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
        )
        if synthesis_sources:
            formatted_ctx, citations = _build_numbered_context(
                synthesis_sources, retrieval_mode="global_synthesis"
            )
            yield ("status", f"Prepared {len(citations)} document samples — generating response…")
            yield ("result", (formatted_ctx, citations))
            return

        logger.warning("[Retrieval:Path A Empty] No ready documents found; falling back to Path B.")

    yield ("status", "Searching Qdrant hybrid index (Dense + BM25)…")
    t_ret = time.perf_counter()
    retriever = build_hybrid_retriever(
        storage=storage,
        embed_model=embed_model,
        user_id=user_id,
        subject=subject,
        doc_id=doc_id,
    )
    leaf_results = retriever.retrieve(expanded_query)
    
    if not leaf_results:
        yield ("result", ("", []))
        return

    yield ("status", f"Matched {len(leaf_results)} leaf nodes — assembling parent chunks…")
    candidates = merge_leaves_to_parents(leaf_results)
    if not candidates:
        yield ("result", ("", []))
        return

    yield ("status", f"Reranking {len(candidates)} candidate passages with Cohere…")
    candidate_texts = [c["text"] for c in candidates]
    reranked = rerank_documents(co, expanded_query, candidate_texts)
    results_list = reranked.results if hasattr(reranked, "results") else reranked

    surviving_sources: list[dict] = []
    for r in results_list:
        rel_score = getattr(r, "relevance_score", None)
        cand = dict(candidates[r.index])
        cand["rerank_score"] = round(float(rel_score), 4) if rel_score is not None else None
        surviving_sources.append(cand)

    if not surviving_sources:
        yield ("result", ("", []))
        return

    final_ctx, citations = _build_numbered_context(
        surviving_sources, retrieval_mode="fact_search"
    )
    yield ("status", f"Selected {len(citations)} verified sources — generating answer…")
    yield ("result", (final_ctx, citations))


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
    _attach_run_metadata(user_id, chat_id, body, target_model)

    summary, recent_messages = prepare_rolling_memory(mongo=mongo, chat_id=chat_id)
    history_for_expansion = []
    if summary:
        history_for_expansion.append({"role": "system", "content": f"Summary of earlier conversation: {summary}"})
    history_for_expansion.extend(recent_messages)

    context, citations = "", []
    for item_type, payload in retrieval_pipeline(
        query=body.query,
        chat_history=history_for_expansion,
        user_id=user_id,
        subject=body.subject,
        doc_id=body.doc_id,
    ):
        if item_type == "result":
            context, citations = payload

    if not context:
        answer = "I couldn't find anything relevant to that in your documents."
        citations = []
    else:
        answer = final_response(
            model_name=target_model,
            query=body.query,
            context=context,
            chat_history=recent_messages,
            summary=summary,
        )
        citations = _mark_cited_sources(answer, citations)

    mongo.add_message(chat_id, user_id, "user", body.query)
    mongo.add_message(chat_id, user_id, "assistant", answer, citations=citations)

    return {
        "messages": [{"role": "assistant", "content": answer}],
        "citations": citations,
    }


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
    _attach_run_metadata(user_id, chat_id, body, target_model)

    meta_payload = json.dumps({
        "type": "meta",
        "chat_id": chat_id,
        "doc_id": body.doc_id,
        "subject": body.subject,
        "model": target_model,
    })
    yield f"data: {meta_payload}\n\n"

    yield f"data: {json.dumps({'type': 'status', 'step': 'Preparing conversation memory…'})}\n\n"
    summary, recent_messages = prepare_rolling_memory(mongo=mongo, chat_id=chat_id)

    history_for_expansion = []
    if summary:
        history_for_expansion.append({"role": "system", "content": f"Summary of earlier conversation: {summary}"})
    history_for_expansion.extend(recent_messages)

    context, citations = "", []
    for item_type, payload in retrieval_pipeline(
        query=body.query,
        chat_history=history_for_expansion,
        user_id=user_id,
        subject=body.subject,
        doc_id=body.doc_id,
    ):
        if item_type == "status":
            yield f"data: {json.dumps({'type': 'status', 'step': payload})}\n\n"
        elif item_type == "result":
            context, citations = payload

    if citations:
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

    full_answer_parts: list[str] = []
    active_streams[chat_id] = False

    if not context:
        fallback = "I couldn't find anything relevant to that in your documents."
        full_answer_parts.append(fallback)
        yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"
    else:
        for token in stream_final_response(
            model_name=target_model,
            query=body.query,
            context=context,
            chat_history=recent_messages,
            summary=summary,
        ):
            if active_streams.get(chat_id):
                stop_msg = "\n\n*[Generation stopped by user]*"
                full_answer_parts.append(stop_msg)
                yield f"data: {json.dumps({'type': 'token', 'content': stop_msg})}\n\n"
                break

            full_answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    active_streams.pop(chat_id, None)

    full_answer = "".join(full_answer_parts)
    citations = _mark_cited_sources(full_answer, citations)

    mongo.add_message(chat_id, user_id, "user", body.query)
    mongo.add_message(chat_id, user_id, "assistant", full_answer, citations=citations)

    yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'citations': citations})}\n\n"