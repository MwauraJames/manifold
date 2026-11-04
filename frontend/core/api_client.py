from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar
from urllib.parse import quote

import requests
import streamlit as st

from core.config import API_URL, ASK_TIMEOUT_S, UPLOAD_TIMEOUT_S

logger = logging.getLogger("manifold.api")

CONNECT_TIMEOUT = 5
STATUS_READ_TIMEOUT = 10
PING_READ_TIMEOUT = 10

_session = requests.Session()

T = TypeVar("T")


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------- dataclasses


@dataclass
class AuthResult:
    user_id: str
    email: str
    access_token: str | None = None
    refresh_token: str | None = None
    confirmation_required: bool = False
    expires_in: int | None = None


@dataclass
class QueuedDocument:
    doc_id: str
    filename: str
    status: str = "queued"


@dataclass
class UploadBatchResult:
    subject: str
    queue_size: int = 0
    documents: list[QueuedDocument] = field(default_factory=list)


@dataclass
class StatusResult:
    doc_id: str
    status: str
    error: str | None = None


@dataclass
class AskResult:
    answer: str
    chat_id: str | None = None
    subject: str | None = None
    doc_id: str | None = None
    sources: list[Any] | None = field(default=None)


@dataclass
class StreamEvent:
    """One Server-Sent Event from POST /ask. kind: meta | token | error | done."""

    kind: str
    data: dict[str, Any]


@dataclass
class ChatSummary:
    chat_id: str
    title: str
    subject: str | None = None
    doc_id: str | None = None
    updated_at: Any = None


@dataclass
class SubjectRecord:
    name: str
    subject_id: str | None = None
    description: str | None = None


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    subject: str
    status: str
    error: str | None = None
    created_at: Any = None
    updated_at: Any = None
    size_bytes: int | None = None


@dataclass
class ModelOption:
    id: str
    name: str
    provider: str = ""


@dataclass
class ModelList:
    default_model: str | None = None
    models: list[ModelOption] = field(default_factory=list)


# ------------------------------------------------------------------- tokens
#
# Tokens live in st.session_state["user"], which is private to ONE browser
# session. (Module-level globals would be shared by every user connected to
# the Streamlit server, so one user's token could end up on another user's
# requests.)


def _user() -> dict[str, Any] | None:
    user = st.session_state.get("user")
    return user if isinstance(user, dict) else None


def _auth_headers() -> dict[str, str]:
    user = _user()
    token = user.get("access_token") if user else None
    return {"Authorization": f"Bearer {token}"} if token else {}


def _set_tokens(access_token: str | None, refresh_token: str | None) -> None:
    user = _user()
    if user is not None:
        user["access_token"] = access_token
        user["refresh_token"] = refresh_token
        st.session_state["user"] = user
        # Refresh tokens rotate, so the copy saved in the browser must be rewritten
        # (core/session_store.py does that at the end of the run).
        st.session_state["_session_dirty"] = True


def _q(value: str) -> str:
    """URL-encode one path segment (subjects, doc ids and chat ids can hold spaces or slashes)."""
    return quote(value, safe="")


# ------------------------------------------------------------------- errors


def _friendly_http_error(response: requests.Response, *, for_status: bool = False) -> ApiError:
    code = response.status_code
    logger.warning("Backend HTTP %s on %s", code, response.url)
    if code == 401:
        return ApiError("Your session expired. Please log in again.", status_code=401)
    if code == 403:
        return ApiError("You are not allowed to access that.", status_code=403)
    if code == 404 and for_status:
        return ApiError("Unknown document.", status_code=404)
    if code == 409:
        return ApiError("That document is still processing.", status_code=409)
    if code == 422:
        return ApiError("The server rejected the request.", status_code=422)
    if 500 <= code <= 599:
        return ApiError("Something went wrong on the server.", status_code=code)
    if code == 404:
        return ApiError("Not found.", status_code=404)
    detail = _error_detail(response)
    return ApiError(detail or "Something went wrong on the server.", status_code=code)


def _error_detail(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    return None


def _map_request_error(exc: Exception) -> ApiError:
    if isinstance(exc, requests.Timeout):
        logger.warning("Backend timeout: %s", type(exc).__name__)
        return ApiError("This is taking too long. Try again.")
    logger.warning("Backend request failed: %s", type(exc).__name__)
    return ApiError("Cannot reach the server. Is the backend running?")


def _once_or_retry(fn: Callable[[], T], retry_connection: bool) -> T:
    try:
        return fn()
    except requests.ConnectionError:
        if retry_connection:
            return fn()
        raise


def _try_refresh() -> bool:
    user = _user()
    refresh_token = user.get("refresh_token") if user else None
    if not refresh_token:
        return False
    try:
        response = _session.request(
            "POST",
            f"{API_URL}/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=(CONNECT_TIMEOUT, STATUS_READ_TIMEOUT),
        )
    except Exception as exc:
        logger.warning("Token refresh failed: %s", type(exc).__name__)
        return False
    if response.status_code >= 400:
        return False
    payload = response.json()
    access = payload.get("access_token")
    # Refresh tokens rotate: always keep the newest one.
    refresh = payload.get("refresh_token") or refresh_token
    if not access:
        return False
    _set_tokens(access, refresh)
    return True


def refresh_with(refresh_token: str) -> AuthResult | None:
    """
    Turn a saved refresh token into a fresh session (used to stay logged in
    across browser refreshes). Returns None if the server rejects the token,
    and raises ApiError if the server can't be reached or fails.
    """
    try:
        response = _session.request(
            "POST",
            f"{API_URL}/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=(CONNECT_TIMEOUT, STATUS_READ_TIMEOUT),
        )
    except Exception as exc:
        raise _map_request_error(exc) from exc
    if 400 <= response.status_code < 500:
        return None
    payload = _json_or_error(response)
    access = payload.get("access_token")
    if not access:
        return None
    return AuthResult(
        user_id=payload.get("user_id") or "",
        email="",
        access_token=access,
        refresh_token=payload.get("refresh_token") or refresh_token,
    )


def _request(
    method: str,
    path: str,
    *,
    authed: bool = True,
    retry_auth: bool = True,
    retry_connection: bool = False,
    timeout: tuple[int, int] = (CONNECT_TIMEOUT, STATUS_READ_TIMEOUT),
    **kwargs: Any,
) -> requests.Response:
    url = f"{API_URL}{path}"
    headers = dict(kwargs.pop("headers", {}) or {})
    if authed:
        headers.update(_auth_headers())

    def _do() -> requests.Response:
        return _session.request(method, url, headers=headers, timeout=timeout, **kwargs)

    try:
        response = _once_or_retry(_do, retry_connection=retry_connection)
    except Exception as exc:
        raise _map_request_error(exc) from exc

    if authed and response.status_code == 401 and retry_auth and _try_refresh():
        headers.update(_auth_headers())
        try:
            response = _do()
        except Exception as exc:
            raise _map_request_error(exc) from exc
    return response


def _json_or_error(response: requests.Response, *, for_status: bool = False) -> Any:
    if response.status_code >= 400:
        raise _friendly_http_error(response, for_status=for_status)
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


# --------------------------------------------------------------------- auth


def signup(email: str, password: str) -> AuthResult:
    response = _request(
        "POST",
        "/auth/signup",
        authed=False,
        json={"email": email, "password": password},
    )
    return _auth_result(_json_or_error(response), email)


def login(email: str, password: str) -> AuthResult:
    response = _request(
        "POST",
        "/auth/login",
        authed=False,
        json={"email": email, "password": password},
    )
    result = _auth_result(_json_or_error(response), email)
    if not result.access_token:
        raise ApiError("Login did not return a session. Try again.")
    return result


def _auth_result(payload: dict[str, Any], fallback_email: str) -> AuthResult:
    return AuthResult(
        user_id=payload.get("user_id") or "",
        email=payload.get("email") or fallback_email,
        access_token=payload.get("access_token"),
        refresh_token=payload.get("refresh_token"),
        confirmation_required=bool(payload.get("confirmation_required")),
        expires_in=payload.get("expires_in"),
    )


# --------------------------------------------------------------------- models


def list_models() -> ModelList:
    response = _request("GET", "/models", retry_connection=True)
    payload = _json_or_error(response)
    options: list[ModelOption] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        # Keep only what the UI needs; any extra server-side fields are dropped here.
        options.append(
            ModelOption(
                id=item["id"],
                name=item.get("name") or item["id"],
                provider=item.get("provider") or "",
            )
        )
    # The backend calls the default "default" (older versions: "default_model").
    default = payload.get("default") or payload.get("default_model")
    if isinstance(default, dict):
        default = default.get("id")
    return ModelList(default_model=default if isinstance(default, str) else None, models=options)


# ------------------------------------------------------------------ documents


def upload_files(files: list[tuple[str, bytes]], subject: str) -> UploadBatchResult:
    """
    Upload several PDFs in ONE request. The backend saves them all, queues
    them, and processes them one after another (status: queued -> processing
    -> ready/failed). `files` is a list of (filename, file_bytes).
    """
    if not files:
        raise ApiError("Choose at least one file.")
    # Repeating the field name "files" is how multipart sends a list.
    multipart = [("files", (name, data, "application/pdf")) for name, data in files]
    response = _request(
        "POST",
        "/upload",
        files=multipart,
        data={"subject": subject},
        # The server saves every file before it answers, so allow more time per file.
        timeout=(CONNECT_TIMEOUT, UPLOAD_TIMEOUT_S + 15 * len(files)),
        retry_auth=True,
        retry_connection=False,
    )
    payload = _json_or_error(response)
    documents = [
        QueuedDocument(
            doc_id=item.get("doc_id") or "",
            filename=item.get("filename") or "untitled",
            status=item.get("status") or "queued",
        )
        for item in (payload.get("documents") or [])
    ]
    return UploadBatchResult(
        subject=payload.get("subject", subject),
        queue_size=int(payload.get("queue_size") or 0),
        documents=documents,
    )


def get_status(doc_id: str) -> StatusResult:
    response = _request("GET", f"/status/{_q(doc_id)}", retry_connection=True)
    payload = _json_or_error(response, for_status=True)
    return StatusResult(
        doc_id=payload.get("doc_id", doc_id),
        status=payload.get("status", "unknown"),
        error=payload.get("error"),
    )


def list_documents(subject: str | None = None) -> list[DocumentRecord]:
    params = {"subject": subject} if subject else None
    response = _request("GET", "/documents", params=params, retry_connection=True)
    payload = _json_or_error(response)
    rows = payload.get("documents") if isinstance(payload, dict) else payload
    result: list[DocumentRecord] = []
    for item in rows or []:
        doc_id = item.get("doc_id") or item.get("id")
        if not doc_id:
            continue
        result.append(
            DocumentRecord(
                doc_id=doc_id,
                filename=item.get("filename") or "untitled",
                subject=item.get("subject") or subject or "",
                status=item.get("status") or "unknown",
                error=item.get("error"),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                size_bytes=item.get("size_bytes"),
            )
        )
    return result


def delete_document(doc_id: str) -> None:
    response = _request("DELETE", f"/documents/{_q(doc_id)}")
    _json_or_error(response)


# ------------------------------------------------------------------- subjects


def list_subjects() -> list[SubjectRecord]:
    response = _request("GET", "/subjects", retry_connection=True)
    payload = _json_or_error(response)
    rows = payload.get("subjects") if isinstance(payload, dict) else payload
    result: list[SubjectRecord] = []
    for item in rows or []:
        name = item.get("name") or item.get("subject")
        if not name:
            continue
        result.append(
            SubjectRecord(
                name=name,
                subject_id=item.get("subject_id"),
                description=item.get("description"),
            )
        )
    return result


def create_subject(name: str, description: str | None = None) -> SubjectRecord:
    payload: dict[str, Any] = {"subject": name}
    if description:
        payload["description"] = description
    response = _request("POST", "/subjects", json=payload)
    item = _json_or_error(response)
    return SubjectRecord(
        name=item.get("name") or name,
        subject_id=item.get("subject_id"),
        description=item.get("description"),
    )


def delete_subject(name: str) -> dict[str, Any]:
    response = _request("DELETE", f"/subjects/{_q(name)}", timeout=(CONNECT_TIMEOUT, ASK_TIMEOUT_S))
    return _json_or_error(response)


# ---------------------------------------------------------------------- chats


def create_chat(
    subject: str | None = None,
    doc_id: str | None = None,
    title: str | None = None,
) -> ChatSummary:
    # Not used by the UI: chats are created by the first /ask so they get a
    # title from the first question. Calling this first freezes the title at
    # "New chat".
    payload: dict[str, Any] = {}
    if subject:
        payload["subject"] = subject
    if doc_id:
        payload["doc_id"] = doc_id
    if title:
        payload["title"] = title
    response = _request("POST", "/chats", json=payload)
    item = _json_or_error(response)
    return ChatSummary(
        chat_id=item.get("chat_id") or "",
        title=item.get("title") or title or "New chat",
        subject=item.get("subject") or subject,
        doc_id=item.get("doc_id") or doc_id,
        updated_at=item.get("updated_at"),
    )


def list_chats(subject: str | None = None) -> list[ChatSummary]:
    params = {"subject": subject} if subject else None
    response = _request("GET", "/chats", params=params, retry_connection=True)
    payload = _json_or_error(response)
    chats = payload.get("chats") or []
    result: list[ChatSummary] = []
    for chat in chats:
        chat_id = chat.get("chat_id")
        if not chat_id:
            continue
        result.append(
            ChatSummary(
                chat_id=chat_id,
                title=chat.get("title") or "New chat",
                subject=chat.get("subject"),
                doc_id=chat.get("doc_id"),
                updated_at=chat.get("updated_at"),
            )
        )
    return result


def get_chat_messages(chat_id: str) -> list[dict[str, Any]]:
    response = _request("GET", f"/chats/{_q(chat_id)}/messages", retry_connection=True)
    payload = _json_or_error(response)
    messages = payload.get("messages") or []
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if content is None:
            content = msg.get("text") or msg.get("message")
        if role not in {"user", "assistant"} or content is None:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def delete_chat(chat_id: str) -> None:
    response = _request("DELETE", f"/chats/{_q(chat_id)}")
    _json_or_error(response)


# ------------------------------------------------------------------------ ask


def ask(
    query: str,
    subject: str,
    doc_id: str | None = None,
    chat_id: str | None = None,
    model: str | None = None,
) -> AskResult:
    """Non-streaming fallback. The UI uses ask_stream()."""
    if not subject:
        raise ApiError("A subject is required.")
    payload: dict[str, Any] = {"query": query, "subject": subject, "stream": False}
    if doc_id:
        payload["doc_id"] = doc_id
    if chat_id:
        payload["chat_id"] = chat_id
    if model:
        payload["model"] = model

    response = _request(
        "POST",
        "/ask",
        json=payload,
        timeout=(CONNECT_TIMEOUT, ASK_TIMEOUT_S),
    )
    body = _json_or_error(response)
    answer = body.get("answer")
    if isinstance(answer, dict):  # some backend versions return {"messages": [{"role", "content"}]}
        last = (answer.get("messages") or [{}])[-1]
        answer = last.get("content") if isinstance(last, dict) else ""
    return AskResult(
        answer=answer or "",
        chat_id=body.get("chat_id") or chat_id,
        subject=body.get("subject") or subject,
        doc_id=body.get("doc_id") or doc_id,
        sources=body.get("sources") or None,
    )


def _parse_sse_event(raw: str) -> StreamEvent | None:
    data_lines = [
        line[5:].lstrip() for line in raw.splitlines() if line.startswith("data:")
    ]
    if not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    return StreamEvent(kind=str(payload.get("type") or ""), data=payload)


def _stream_events(path: str, payload: dict[str, Any], token: str | None = None) -> Iterator[StreamEvent]:
    """
    POST `payload` to `path` and yield the Server-Sent Events as they arrive.

    Events: meta (chat_id, doc_id, subject, model), token (content), done
    (chat_id). Errors before the stream starts (bad model, wrong chat owner,
    expired session) are raised as ApiError.

    Pass `token` when calling from a background thread: Streamlit's session
    state is not available there, so the token is supplied explicitly and no
    automatic refresh is attempted.
    """

    def _headers() -> dict[str, str]:
        base = {"Authorization": f"Bearer {token}"} if token else _auth_headers()
        return {**base, "Accept": "text/event-stream"}

    def _open() -> requests.Response:
        return _session.post(
            f"{API_URL}{path}",
            json=payload,
            headers=_headers(),
            stream=True,
            timeout=(CONNECT_TIMEOUT, ASK_TIMEOUT_S),
        )

    try:
        response = _open()
        if token is None and response.status_code == 401 and _try_refresh():
            response.close()
            response = _open()
    except Exception as exc:
        raise _map_request_error(exc) from exc

    if response.status_code >= 400:
        error = _friendly_http_error(response)
        response.close()
        raise error

    response.encoding = "utf-8"
    buffer = ""
    started = False   # have we received at least one event?
    saw_done = False  # did the server finish the answer?
    try:
        # chunk_size=None yields data as each network chunk arrives. (iter_lines
        # would wait for 512 bytes, which delays short answers.)
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            # A chunk can end mid-event: only parse complete events.
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                event = _parse_sse_event(raw)
                if event:
                    started = True
                    saw_done = saw_done or event.kind == "done"
                    yield event
        if buffer.strip():
            event = _parse_sse_event(buffer)
            if event:
                started = True
                saw_done = saw_done or event.kind == "done"
                yield event
        # The backend saves a turn only after the answer is complete, and it sends
        # no "error" event, so a stream that stops without "done" was cut off.
        if started and not saw_done:
            raise ApiError("The connection closed before the answer finished.")
    except requests.RequestException as exc:
        if started:
            raise ApiError("The answer was interrupted before it finished.") from exc
        raise _map_request_error(exc) from exc
    finally:
        response.close()


def ask_stream(
    query: str,
    subject: str,
    doc_id: str | None = None,
    chat_id: str | None = None,
    model: str | None = None,
    token: str | None = None,
) -> Iterator[StreamEvent]:
    """POST /ask with stream=true."""
    if not subject:
        raise ApiError("A subject is required.")
    payload: dict[str, Any] = {"query": query, "subject": subject, "stream": True}
    if doc_id:
        payload["doc_id"] = doc_id
    if chat_id:
        payload["chat_id"] = chat_id
    if model:
        payload["model"] = model
    return _stream_events("/ask", payload, token)


def edit_last_stream(
    chat_id: str,
    query: str,
    model: str | None = None,
    token: str | None = None,
) -> Iterator[StreamEvent]:
    """
    POST /chats/{chat_id}/edit_last: the backend deletes the chat's last
    question and answer, then streams a new answer to the edited question.
    The chat's subject and document scope are taken from the chat itself.
    """
    payload: dict[str, Any] = {"query": query, "stream": True}
    if model:
        payload["model"] = model
    return _stream_events(f"/chats/{_q(chat_id)}/edit_last", payload, token)


def stop_generation(chat_id: str) -> None:
    """Ask the backend to stop the answer currently streaming for this chat."""
    response = _request("POST", f"/chats/{_q(chat_id)}/stop")
    _json_or_error(response)


def ensure_session() -> str:
    """
    Cheap authenticated call that refreshes an expired access token (in the
    main thread, where session state exists) before a background thread
    starts using it. Raises ApiError(401) if the session can't be refreshed.
    """
    response = _request("GET", "/auth/me", retry_connection=True)
    return _json_or_error(response).get("user_id") or ""


def current_access_token() -> str | None:
    user = _user()
    return user.get("access_token") if user else None


def ping() -> bool:
    try:
        _request("GET", "/", authed=False, retry_connection=True, timeout=(CONNECT_TIMEOUT, PING_READ_TIMEOUT))
        return True
    except Exception as exc:
        logger.warning("Backend ping failed: %s", type(exc).__name__)
        return False
