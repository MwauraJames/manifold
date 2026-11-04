from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from core import api_client
from core.api_client import ApiError

logger = logging.getLogger("manifold.stream")


@dataclass
class JobSnapshot:
    state: str                      # running | done | failed
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    error: ApiError | None = None


class StreamJob:
    """
    Reads one streamed answer on a background thread.

    Why a thread: a Streamlit button click reruns the script, which would kill a
    stream being read in the script itself. The backend's Stop works by keeping
    the connection open until it sends a final "stopped" token and "done", so the
    stream must survive reruns. The UI polls snapshot() instead.

    The thread never touches st.session_state; it only writes to this object.
    """

    def __init__(
        self,
        *,
        kind: str,                  # "ask" or "edit"
        prompt: str,
        subject: str,
        doc_id: str | None,
        chat_id: str | None,
        model: str | None,
        view_id: str,
    ) -> None:
        self.kind = kind
        self.prompt = prompt
        self.subject = subject
        self.doc_id = doc_id
        self.chat_id = chat_id
        self.model = model
        self.view_id = view_id
        self.stop_requested = False
        self._lock = threading.Lock()
        self._text = ""
        self._meta: dict[str, Any] = {}
        self._state = "running"
        self._error: ApiError | None = None
        self._thread: threading.Thread | None = None

    def start(self, token: str | None) -> None:
        self._thread = threading.Thread(target=self._run, args=(token,), daemon=True)
        self._thread.start()

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(self._state, self._text, dict(self._meta), self._error)

    def _run(self, token: str | None) -> None:
        try:
            if self.kind == "edit":
                events = api_client.edit_last_stream(
                    self.chat_id or "", self.prompt, model=self.model, token=token
                )
            else:
                events = api_client.ask_stream(
                    self.prompt,
                    self.subject,
                    doc_id=self.doc_id,
                    chat_id=self.chat_id,
                    model=self.model,
                    token=token,
                )
            for event in events:
                with self._lock:
                    if event.kind == "meta":
                        self._meta.update(event.data)
                    elif event.kind == "token":
                        self._text += event.data.get("content") or ""
            with self._lock:
                self._state = "done"
        except ApiError as exc:
            with self._lock:
                self._error, self._state = exc, "failed"
        except Exception:  # never let the thread die silently
            logger.exception("Unexpected error while streaming an answer")
            with self._lock:
                self._error = ApiError("Something went wrong while reading the answer.")
                self._state = "failed"
