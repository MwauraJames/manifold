from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MAX_SUBJECT_LEN = 60
RESERVED_SUBJECT = "string"  # the backend treats this literal text as "not provided"

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")

# A document is "active" while the server still has work to do on it.
ACTIVE_STATUSES = {"queued", "processing"}


def slug(value: str) -> str:
    """Mirror of the backend's safe_slug(). Used to predict doc_id clashes."""
    cleaned = _SLUG_RE.sub("_", (value or "").strip()).strip("._")
    return cleaned or "unnamed"


def parse_server_time(value: Any) -> datetime | None:
    """
    The backend sends UTC timestamps WITHOUT a timezone suffix
    (e.g. '2026-10-28T10:15:30.123000'). Treat them as UTC explicitly,
    otherwise they get read as local time and show the wrong hour.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_date(value: Any) -> str:
    dt = parse_server_time(value)
    if dt is None:
        return ""
    return dt.astimezone().strftime("%d %b %Y, %H:%M")


def minutes_since(value: Any) -> float | None:
    dt = parse_server_time(value)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60


def validate_subject_name(name: str, existing: list[str]) -> str | None:
    cleaned = (name or "").strip()
    if not cleaned:
        return "Enter a subject name."
    if len(cleaned) > MAX_SUBJECT_LEN:
        return f"Keep the name under {MAX_SUBJECT_LEN} characters."
    if cleaned.lower() == RESERVED_SUBJECT:
        return "That name is reserved. Pick another."
    for other in existing:
        if other.lower() == cleaned.lower():
            return "You already have a subject with that name."
        if slug(other) == slug(cleaned):
            return f"Too similar to the existing subject '{other}'."
    return None


@dataclass
class UploadPlan:
    accepted: list[Any] = field(default_factory=list)          # files to send
    replacing: list[str] = field(default_factory=list)         # filenames that overwrite an existing document
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    error: str | None = None                                   # blocks the whole upload


def plan_upload(files: list[Any], existing_docs: list[Any], max_mb: int, max_files: int) -> UploadPlan:
    """
    Decide what can be sent. `files` need .name and .size (Streamlit's
    UploadedFile); `existing_docs` need .filename and .status.

    The server names a document after the slug of its filename, so two files
    with the same slug (in this batch or already in the subject) are the same
    document: the later one would overwrite the earlier one.
    """
    plan = UploadPlan()
    if len(files) > max_files:
        plan.error = f"Select at most {max_files} files at a time (you chose {len(files)})."
        return plan

    existing = {slug(d.filename): d for d in existing_docs}
    seen: set[str] = set()
    for f in files:
        key = slug(f.name)
        reason = None
        if not f.name.lower().endswith((".pdf", ".doc", ".docx", ".pptx",".md",".txt")):
            reason = "unsupported file type"
        elif f.size == 0:
            reason = "the file is empty"
        elif f.size > max_mb * 1024 * 1024:
            reason = f"larger than {max_mb} MB"
        elif key in seen:
            reason = "another selected file has the same name"
        elif key in existing and existing[key].status in ACTIVE_STATUSES:
            reason = "a document with this name is still being processed"

        if reason:
            plan.skipped.append((f.name, reason))
            continue
        seen.add(key)
        plan.accepted.append(f)
        if key in existing:
            plan.replacing.append(f.name)
    return plan
