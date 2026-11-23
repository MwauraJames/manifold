# app/ingestion/mongo_store.py
import os
import math
import re
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime, timezone
from typing import Optional
from bson.objectid import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
import gridfs

from configs.config import Settings

# Always load app/.env reliably regardless of current working directory
_APP_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_APP_ENV, override=True)
load_dotenv(override=True)

settings = Settings()


def _resolve_mongo_uri() -> str:
    """
    Reads MONGODB_URI (or MONGO_URI) from .env and safely injects
    MONGODB_USERNAME and MONGODB_PASSWORD if placeholders are present.
    """
    raw_uri = (
        os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URI")
        or getattr(settings, "mongo_uri", "mongodb://localhost:27017")
    ).strip()

    username = (os.getenv("MONGODB_USERNAME") or "").strip()
    password = (os.getenv("MONGODB_PASSWORD") or "").strip()

    # If the Atlas URI still has <db_password> or <password> placeholders, replace them
    if password:
        encoded_pw = quote_plus(password)
        for placeholder in ("<db_password>", "<password>", "${MONGODB_PASSWORD}"):
            if placeholder in raw_uri:
                raw_uri = raw_uri.replace(placeholder, encoded_pw)

    if username:
        encoded_user = quote_plus(username)
        for placeholder in ("<db_username>", "<username>", "${MONGODB_USERNAME}"):
            if placeholder in raw_uri:
                raw_uri = raw_uri.replace(placeholder, encoded_user)

    return raw_uri


class MongoStore:
    def __init__(self):
        mongo_uri = _resolve_mongo_uri()
        mongo_db_name = (
            os.getenv("MONGODB_DB_NAME")
            or os.getenv("MONGO_DB_NAME")
            or getattr(settings, "mongo_db_name", "manifold")
        )

        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_db_name]
        self.fs = gridfs.GridFS(self.db)

        self.documents = self.db["documents"]
        self.chats = self.db["chats"]
        self.messages = self.db["messages"]
        self.subjects = self.db["subjects"]

        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for fast lookup by user_id, subject, doc_id, and chat_id."""
        self.documents.create_index([("doc_id", ASCENDING)], unique=True)
        self.documents.create_index([("user_id", ASCENDING), ("subject", ASCENDING)])

        self.chats.create_index([("chat_id", ASCENDING)], unique=True)
        self.chats.create_index([("user_id", ASCENDING), ("subject", ASCENDING)])

        self.messages.create_index([("chat_id", ASCENDING), ("created_at", ASCENDING)])

        # Unique index per (user_id, name) on subjects
        self.subjects.create_index(
            [("user_id", ASCENDING), ("name", ASCENDING)], unique=True
        )

    # ---------------- Document Storage (Raw + Parsed) ----------------

    # Inside MongoStore in app/ingestion/mongo_store.py

    def init_document_status(
        self,
        doc_id: str,
        user_id: str,
        subject: str,
        filename: str,
        status: str = "queued",
    ):
        """Fast (<1ms) metadata upsert when upload enters the queue."""
        now = datetime.now(timezone.utc)
        self.documents.update_one(
            {"doc_id": doc_id},
            {
                "$set": {
                    "user_id": user_id,
                    "subject": subject,
                    "filename": filename,
                    "status": status,
                    "error": None,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def set_document_processing(self, doc_id: str):
        """Mark document as actively processing when popped from the queue."""
        self.documents.update_one(
            {"doc_id": doc_id},
            {
                "$set": {
                    "status": "processing",
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    def finalize_document(
        self,
        doc_id: str,
        user_id: str,
        subject: str,
        filename: str,
        file_path: str,
        content_type: str,
        parsed_markdown: str,
        num_leaves: int,
    ):
        """
        Runs AFTER Qdrant ingestion finishes:
          1. Streams the raw file from local `file_path` into MongoDB GridFS.
          2. Saves `parsed_markdown` and marks the document status as 'ready'.
        """
        existing = self.documents.find_one({"doc_id": doc_id})
        if existing and existing.get("gridfs_id"):
            try:
                self.fs.delete(ObjectId(existing["gridfs_id"]))
            except Exception:
                pass

        # Stream directly from the local file into GridFS
        with open(file_path, "rb") as f:
            gridfs_id = self.fs.put(
                f,
                filename=filename,
                user_id=user_id,
                subject=subject,
                doc_id=doc_id,
                content_type=content_type,
            )

        self.documents.update_one(
            {"doc_id": doc_id},
            {
                "$set": {
                    "gridfs_id": str(gridfs_id),
                    "parsed_markdown": parsed_markdown,
                    "num_leaves": num_leaves,
                    "status": "ready",
                    "error": None,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    def mark_document_failed(self, doc_id: str, error_msg: str):
        """Record ingestion failure reason in MongoDB."""
        self.documents.update_one(
            {"doc_id": doc_id},
            {
                "$set": {
                    "status": "failed",
                    "error": error_msg,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    def get_document_status(self, doc_id: str) -> Optional[dict]:
        """Fetch document status (excludes heavy parsed_markdown field for speed)."""
        return self.documents.find_one(
            {"doc_id": doc_id},
            {"_id": 0, "parsed_markdown": 0},
        )

    def get_raw_file(self, doc_id: str) -> Optional[bytes]:
        """Retrieve the original unprocessed file bytes from GridFS."""
        doc = self.documents.find_one({"doc_id": doc_id}, {"gridfs_id": 1})
        if not doc or not doc.get("gridfs_id"):
            return None
        return self.fs.get(ObjectId(doc["gridfs_id"])).read()

    def list_user_documents(
        self, user_id: str, subject: Optional[str] = None
    ) -> list[dict]:
        """List all uploaded documents for a user (optionally filtered by subject)."""
        query = {"user_id": user_id}
        if subject:
            query["subject"] = subject
        return list(
            self.documents.find(
                query, {"_id": 0, "parsed_markdown": 0}
            ).sort("created_at", DESCENDING)
        )

    # ---------------- Continuous Chat History ----------------

    def ensure_chat_session(
        self,
        chat_id: str,
        user_id: str,
        subject: Optional[str] = None,
        doc_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> dict:
        """Create a chat session if it doesn't exist yet, optionally scoped to a doc_id."""
        now = datetime.now(timezone.utc)
        self.chats.update_one(
            {"chat_id": chat_id},
            {
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "user_id": user_id,
                    "subject": subject,
                    "doc_id": doc_id,
                    "title": title or "New Study Session",
                    "created_at": now,
                },
            },
            upsert=True,
        )
        return self.chats.find_one({"chat_id": chat_id}, {"_id": 0})

    def get_chat_session(self, chat_id: str) -> Optional[dict]:
        """Fetch a single chat session's metadata (subject, doc_id, user_id)."""
        return self.chats.find_one({"chat_id": chat_id}, {"_id": 0})

    def update_chat_doc_id(self, chat_id: str, doc_id: Optional[str]) -> None:
        """Persist an updated document scope (or None for all documents) on an existing chat."""
        self.chats.update_one(
            {"chat_id": chat_id},
            {
                "$set": {
                    "doc_id": doc_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    def list_user_chats(
        self, user_id: str, subject: Optional[str] = None
    ) -> list[dict]:
        """Return all chat sessions for a user, most recently updated first."""
        query = {"user_id": user_id}
        if subject:
            query["subject"] = subject
        return list(
            self.chats.find(query, {"_id": 0}).sort("updated_at", DESCENDING)
        )

    # Inside MongoStore in app/ingestion/mongo_store.py

    def get_filenames_by_doc_ids(self, doc_ids: list[str]) -> dict[str, str]:
        """Batch lookup mapping doc_id -> filename for backfilling legacy Qdrant nodes."""
        clean_ids = [d for d in set(doc_ids) if d]
        if not clean_ids:
            return {}
        cursor = self.documents.find(
            {"doc_id": {"$in": clean_ids}},
            {"_id": 0, "doc_id": 1, "filename": 1},
        )
        return {
            doc["doc_id"]: doc.get("filename") or doc["doc_id"]
            for doc in cursor
            if doc.get("doc_id")
        }

    def add_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        citations: Optional[list[dict]] = None,
    ) -> dict:
        """Append a message ('user' or 'assistant') to a chat session and bump chat updated_at."""
        now = datetime.now(timezone.utc)
        msg_doc = {
            "chat_id": chat_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": now,
        }
        if citations is not None:
            msg_doc["citations"] = citations

        self.messages.insert_one(msg_doc)
        self.chats.update_one(
            {"chat_id": chat_id},
            {"$set": {"updated_at": now}},
        )
        msg_doc.pop("_id", None)
        return msg_doc

    def get_recent_messages(
        self,
        chat_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Fetch the most recent `limit` messages for a chat, returned in
        chronological order (oldest -> newest) for multi-turn LLM context.
        """
        cursor = (
            self.messages.find({"chat_id": chat_id}, {"_id": 0})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        recent = list(cursor)
        recent.reverse()
        return recent

    # ---------------- Subject Management ----------------

    def ensure_subject(
        self, user_id: str, subject: str, description: Optional[str] = None
    ) -> dict:
        """Create or update a subject record for a user."""
        now = datetime.now(timezone.utc)
        update_doc = {
            "$set": {"updated_at": now},
            "$setOnInsert": {
                "subject_id": str(ObjectId()),
                "user_id": user_id,
                "name": subject,
                "created_at": now,
            },
        }
        if description is not None:
            update_doc["$set"]["description"] = description
        else:
            update_doc["$setOnInsert"]["description"] = ""

        self.subjects.update_one(
            {"user_id": user_id, "name": subject},
            update_doc,
            upsert=True,
        )
        return self.subjects.find_one(
            {"user_id": user_id, "name": subject}, {"_id": 0}
        )

    def list_user_subjects(self, user_id: str) -> list[dict]:
        """
        Return all subjects for a user. Also backfills any legacy subjects
        that only existed as strings on `documents` or `chats`.
        """
        stored_subjects = list(
            self.subjects.find({"user_id": user_id}, {"_id": 0}).sort(
                "created_at", DESCENDING
            )
        )
        known_names = {s["name"] for s in stored_subjects}

        # Discover any subjects attached only to existing docs or chats
        doc_subjects = self.documents.distinct("subject", {"user_id": user_id})
        chat_subjects = self.chats.distinct("subject", {"user_id": user_id})

        for name in set(doc_subjects) | set(chat_subjects):
            if name and name not in known_names:
                created = self.ensure_subject(user_id=user_id, subject=name)
                stored_subjects.append(created)
                known_names.add(name)

        return stored_subjects

    # ---------------- Deletion Methods ----------------

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a single document's raw PDF from GridFS and its record
        (including parsed_markdown) from the `documents` collection.
        """
        doc = self.documents.find_one({"doc_id": doc_id})
        if not doc:
            return False

        if doc.get("gridfs_id"):
            try:
                self.fs.delete(ObjectId(doc["gridfs_id"]))
            except Exception:
                pass

        self.documents.delete_one({"doc_id": doc_id})
        return True

    def delete_chat(self, chat_id: str) -> bool:
        """Deletes a chat session and all of its messages."""
        res = self.chats.delete_one({"chat_id": chat_id})
        self.messages.delete_many({"chat_id": chat_id})
        return res.deleted_count > 0

    def delete_subject(self, user_id: str, subject: str) -> dict:
        """
        Deletes EVERYTHING in MongoDB related to (user_id, subject):
          1. Raw PDF binaries in GridFS
          2. Parsed documents & status records in `documents`
          3. All chat sessions in `chats` for this subject
          4. All chat messages in `messages` belonging to those chats
        """
        # 1. Delete all GridFS raw files and document records for this subject
        docs = list(
            self.documents.find(
                {"user_id": user_id, "subject": subject},
                {"doc_id": 1, "gridfs_id": 1},
            )
        )
        for doc in docs:
            if doc.get("gridfs_id"):
                try:
                    self.fs.delete(ObjectId(doc["gridfs_id"]))
                except Exception:
                    pass
        doc_del = self.documents.delete_many(
            {"user_id": user_id, "subject": subject}
        )

        # 2. Find all chat_ids for this subject and delete their messages + sessions
        subject_chats = list(
            self.chats.find(
                {"user_id": user_id, "subject": subject},
                {"chat_id": 1},
            )
        )
        chat_ids = [c["chat_id"] for c in subject_chats]

        msg_deleted = 0
        if chat_ids:
            msg_res = self.messages.delete_many({"chat_id": {"$in": chat_ids}})
            msg_deleted = msg_res.deleted_count

        chat_res = self.chats.delete_many(
            {"user_id": user_id, "subject": subject}
        )

        # 3. Delete the subject record itself from `subjects`
        subj_res = self.subjects.delete_one({"user_id": user_id, "name": subject})

        return {
            "deleted_subject": subj_res.deleted_count,
            "deleted_documents": doc_del.deleted_count,
            "deleted_chats": chat_res.deleted_count,
            "deleted_messages": msg_deleted,
            "doc_ids": [d["doc_id"] for d in docs],
        }
# Add inside MongoStore in app/ingestion/mongo_store.py

    def get_chat_memory_state(self, chat_id: str) -> dict:
        """
        Returns the rolling summary, how many messages are already summarized,
        and the list of unsummarized messages in chronological order.
        """
        chat = self.chats.find_one(
            {"chat_id": chat_id},
            {"_id": 0, "summary": 1, "summarized_count": 1},
        ) or {}

        summary = chat.get("summary")
        summarized_count = chat.get("summarized_count", 0)

        # Fetch only the messages that occurred AFTER the last summary cutoff
        cursor = (
            self.messages.find(
                {"chat_id": chat_id},
                {"_id": 0, "role": 1, "content": 1, "created_at": 1},
            )
            .sort("created_at", ASCENDING)
            .skip(summarized_count)
        )
        unsummarized_messages = list(cursor)

        return {
            "summary": summary,
            "summarized_count": summarized_count,
            "unsummarized_messages": unsummarized_messages,
        }

    def update_chat_summary(
        self, chat_id: str, new_summary: str, new_summarized_count: int
    ):
        """Replaces the previous summary with the new summary and advances the pointer."""
        self.chats.update_one(
            {"chat_id": chat_id},
            {
                "$set": {
                    "summary": new_summary,
                    "summarized_count": new_summarized_count,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    # ---------------- Stratified Synthesis Context (Adaptive RAG) ----------------

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    _LOW_SIGNAL_HEADING_RE = re.compile(
        r"^(references|bibliography|works\s+cited|table\s+of\s+contents|contents|acknowledg(e)?ments|index)$",
        re.IGNORECASE,
    )

    @classmethod
    def _split_markdown_units(cls, text: str, max_unit_chars: int = 2000) -> list[str]:
        """
        Splits Markdown into atomic units without breaking fenced code blocks,
        display LaTeX ($$...$$), or contiguous Markdown tables.
        """
        lines = text.splitlines()
        units: list[str] = []
        buf: list[str] = []
        in_fence = False
        fence_marker = ""
        in_math = False
        in_table = False

        def flush():
            if buf:
                joined = "\n".join(buf).strip()
                if joined:
                    units.append(joined)
                buf.clear()

        for line in lines:
            stripped = line.strip()

            # 1. Track fenced code blocks (``` or ~~~)
            if not in_math and (stripped.startswith("```") or stripped.startswith("~~~")):
                marker = stripped[:3]
                if not in_fence:
                    if in_table:
                        flush()
                        in_table = False
                    flush()
                    in_fence = True
                    fence_marker = marker
                    buf.append(line)
                    continue
                elif stripped.startswith(fence_marker):
                    buf.append(line)
                    flush()
                    in_fence = False
                    fence_marker = ""
                    continue

            if in_fence:
                buf.append(line)
                # Safety valve for malformed unclosed fences
                if len(buf) > 120 or sum(len(x) for x in buf) > max_unit_chars * 2.5:
                    flush()
                    in_fence = False
                    fence_marker = ""
                continue

            # 2. Track multi-line display LaTeX ($$...$$)
            if stripped.count("$$") % 2 == 1:
                if not in_math:
                    if in_table:
                        flush()
                        in_table = False
                    flush()
                    in_math = True
                    buf.append(line)
                    continue
                else:
                    buf.append(line)
                    flush()
                    in_math = False
                    continue

            if in_math:
                buf.append(line)
                # Safety valve for stray unclosed $$ in noisy OCR
                if len(buf) > 35 or sum(len(x) for x in buf) > max_unit_chars * 1.5:
                    flush()
                    in_math = False
                continue

            # 3. Track contiguous Markdown tables
            is_table_line = stripped.startswith("|") and "|" in stripped[1:]
            if is_table_line:
                if not in_table:
                    flush()
                    in_table = True
                buf.append(line)
                continue
            elif in_table:
                flush()
                in_table = False

            # 4. Blank lines flush paragraphs
            if not stripped:
                flush()
                continue

            # 5. Headings always start a fresh unit
            if cls._HEADING_RE.match(stripped):
                flush()
                buf.append(line)
                flush()
                continue

            buf.append(line)

        flush()

        # Sub-split any pathological giant plain-text paragraph (e.g. raw OCR wall of text)
        normalized: list[str] = []
        for u in units:
            if len(u) <= max_unit_chars:
                normalized.append(u)
                continue

            is_structured = (
                u.startswith("```")
                or u.startswith("~~~")
                or u.startswith("$$")
                or u.startswith("|")
            )
            if is_structured and len(u) <= int(max_unit_chars * 1.5):
                normalized.append(u)
                continue

            # Split oversized units on sentences or newlines near max_unit_chars
            remaining = u
            while len(remaining) > max_unit_chars:
                cut = remaining.rfind("\n", 0, max_unit_chars)
                if cut < int(max_unit_chars * 0.5):
                    sentence_matches = list(
                        re.finditer(r"[.!?]\s+", remaining[:max_unit_chars])
                    )
                    cut = sentence_matches[-1].end() if sentence_matches else max_unit_chars
                normalized.append(remaining[:cut].strip())
                remaining = remaining[cut:].strip()
            if remaining:
                normalized.append(remaining)

        return normalized

    @classmethod
    def _build_markdown_blocks(
        cls,
        text: str,
        target_chars: int = 2000,
    ) -> list[dict]:
        """
        Groups atomic Markdown units into ~512-token (~2,000-char) blocks while
        tracking active section headings, positional percentages, and low-signal flags.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return []

        units = cls._split_markdown_units(clean_text, max_unit_chars=target_chars)
        if not units:
            return []

        blocks: list[dict] = []
        current_units: list[str] = []
        current_len = 0
        active_heading = ""
        block_heading = ""
        total_len = max(1, len(clean_text))
        cursor = 0
        block_start = 0

        def finalize_block():
            nonlocal current_units, current_len, block_heading, block_start
            if not current_units:
                return
            blk_text = "\n\n".join(current_units).strip()
            if not blk_text:
                current_units = []
                current_len = 0
                return

            heading_clean = block_heading.strip()
            alnum_count = len(re.findall(r"[A-Za-z0-9]", blk_text))
            is_low_signal = (
                bool(heading_clean and cls._LOW_SIGNAL_HEADING_RE.match(heading_clean))
                or alnum_count < 120
            )

            pct_start = max(0, min(100, int((block_start / total_len) * 100)))
            pct_end = max(pct_start, min(100, int((cursor / total_len) * 100)))

            blocks.append(
                {
                    "text": blk_text,
                    "heading": heading_clean,
                    "pct_start": pct_start,
                    "pct_end": pct_end,
                    "is_low_signal": is_low_signal,
                }
            )
            current_units = []
            current_len = 0
            block_heading = active_heading
            block_start = cursor

        for u in units:
            heading_match = cls._HEADING_RE.match(u.strip())
            if heading_match:
                new_heading = heading_match.group(2).strip()
                # Start a fresh block at a heading if the current block already has body content
                if current_units and current_len >= int(target_chars * 0.45):
                    finalize_block()
                active_heading = new_heading
                if not current_units:
                    block_heading = new_heading
                    block_start = cursor
                elif not block_heading:
                    block_heading = new_heading

                current_units.append(u)
                current_len += len(u) + 2
                cursor += len(u) + 2
                continue

            if current_units and (current_len + len(u) > target_chars):
                finalize_block()

            if not current_units:
                block_heading = active_heading
                block_start = cursor

            current_units.append(u)
            current_len += len(u) + 2
            cursor += len(u) + 2

        finalize_block()
        return blocks

    @classmethod
    def _extract_heading_spine(
        cls,
        docs: list[dict],
        max_chars: int = 1500,
    ) -> str:
        """
        Builds a compact per-document table of contents from Markdown headings
        so global synthesis sees the complete topic map across all uploaded files.
        """
        if not docs or max_chars <= 0:
            return ""

        per_doc_cap = max(120, max_chars // len(docs))
        lines_out: list[str] = []

        for d in docs:
            fname = d.get("filename") or d.get("doc_id") or "Document"
            md = d.get("parsed_markdown") or ""
            primary_headings: list[str] = []
            h3_headings: list[str] = []
            in_fence = False

            for raw_line in md.splitlines():
                stripped = raw_line.strip()
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                m = cls._HEADING_RE.match(stripped)
                if not m:
                    continue
                level = len(m.group(1))
                title = re.sub(r"\s+", " ", m.group(2)).strip(" #:*_`")
                if not title or len(title) > 90:
                    continue
                if level <= 2 and title not in primary_headings:
                    primary_headings.append(title)
                elif level == 3 and title not in h3_headings:
                    h3_headings.append(title)

            headings = primary_headings if len(primary_headings) >= 3 else (primary_headings + h3_headings)
            if not headings:
                continue

            prefix = f"- {fname}: "
            joined = " > ".join(headings)
            entry = prefix + joined
            if len(entry) > per_doc_cap:
                entry = entry[: max(len(prefix) + 20, per_doc_cap - 1)].rstrip(" >") + "…"
            lines_out.append(entry)

        spine = "\n".join(lines_out)
        if len(spine) > max_chars:
            spine = spine[: max_chars - 1].rstrip() + "…"
        return spine

    @staticmethod
    def _allocate_block_slots(
        doc_lengths: list[int],
        doc_block_counts: list[int],
        total_slots: int,
    ) -> list[int]:
        """
        Stratified slot allocation across multiple documents:
          1. Guarantees every document at least 1 block (no silent drops).
          2. Distributes remaining slots proportionally to sqrt(doc_length)
             while respecting each document's actual block count.
        """
        n = len(doc_lengths)
        if n == 0:
            return []

        alloc = [min(1, c) for c in doc_block_counts]
        remaining = total_slots - sum(alloc)
        if remaining <= 0:
            return alloc

        weights = [math.sqrt(max(1, length)) for length in doc_lengths]

        while remaining > 0:
            eligible = [i for i in range(n) if alloc[i] < doc_block_counts[i]]
            if not eligible:
                break

            weight_sum = sum(weights[i] for i in eligible)
            if weight_sum <= 0:
                break

            exact_shares = {i: remaining * (weights[i] / weight_sum) for i in eligible}
            added_this_round = 0

            for i in eligible:
                floor_add = min(int(exact_shares[i]), doc_block_counts[i] - alloc[i])
                if floor_add > 0:
                    alloc[i] += floor_add
                    remaining -= floor_add
                    added_this_round += floor_add

            if remaining <= 0:
                break

            still_eligible = [i for i in eligible if alloc[i] < doc_block_counts[i]]
            if not still_eligible:
                break

            still_eligible.sort(
                key=lambda i: (exact_shares[i] - int(exact_shares[i]), weights[i]),
                reverse=True,
            )
            for i in still_eligible:
                if remaining <= 0:
                    break
                alloc[i] += 1
                remaining -= 1
                added_this_round += 1

            if added_this_round == 0:
                break

        return alloc

    @staticmethod
    def _sample_doc_blocks(
        blocks: list[dict],
        k: int,
        max_block_chars: int,
    ) -> list[dict]:
        """
        Divides a document's blocks into `k` equal sequential segments and picks
        the best block near each segment's midpoint, avoiding low-signal cover/reference
        blocks whenever substantive content exists in that segment.
        """
        n = len(blocks)
        if n == 0 or k <= 0:
            return []

        if n <= k:
            selected = blocks
        else:
            selected = []
            used_indices: set[int] = set()
            for seg_idx in range(k):
                seg_start = int(seg_idx * n / k)
                seg_end = max(seg_start + 1, int((seg_idx + 1) * n / k))
                candidates = [idx for idx in range(seg_start, seg_end) if idx not in used_indices]
                if not candidates:
                    continue

                midpoint = (seg_start + seg_end - 1) / 2.0
                substantive = [idx for idx in candidates if not blocks[idx]["is_low_signal"]]
                pool = substantive if substantive else candidates

                best_idx = min(
                    pool,
                    key=lambda idx: (abs(idx - midpoint), -len(blocks[idx]["text"])),
                )
                used_indices.add(best_idx)
                selected.append(blocks[best_idx])

        trimmed: list[dict] = []
        for blk in selected:
            text = blk["text"]
            if len(text) > max_block_chars:
                cut = text.rfind("\n\n", 0, max_block_chars)
                if cut < int(max_block_chars * 0.5):
                    cut = text.rfind("\n", 0, max_block_chars)
                if cut < int(max_block_chars * 0.5):
                    cut = max_block_chars
                text = text[:cut].rstrip() + "\n..."
            item = dict(blk)
            item["text"] = text
            trimmed.append(item)

        return trimmed

    def get_synthesis_bundle(
        self,
        user_id: str,
        subject: Optional[str] = None,
        doc_id: Optional[str] = None,
        max_total_chars: Optional[int] = None,
        target_block_chars: int = 2000,
        max_outline_chars: int = 1500,
    ) -> tuple[list[dict], str]:
        """
        Returns (sources, heading_outline) for global_synthesis:
          - Short-circuits to 100% of content when total markdown <= budget.
          - Otherwise uses Stratified Uniform Sampling (floor + sqrt(length) slots)
            over paragraph/heading-aligned ~512-token blocks + a full heading spine.
        """
        budget = max_total_chars or getattr(settings, "synthesis_max_chars", 24000)

        query: dict = {"user_id": user_id, "status": "ready"}
        if doc_id:
            query["doc_id"] = doc_id
        elif subject:
            query["subject"] = subject

        docs = list(
            self.documents.find(
                query,
                {"_id": 0, "doc_id": 1, "filename": 1, "subject": 1, "parsed_markdown": 1},
            ).sort("created_at", ASCENDING)
        )

        valid_docs = [d for d in docs if (d.get("parsed_markdown") or "").strip()]
        if not valid_docs:
            return [], ""

        heading_outline = self._extract_heading_spine(valid_docs, max_chars=max_outline_chars)
        total_chars = sum(len(d["parsed_markdown"].strip()) for d in valid_docs)

        doc_blocks_list = [
            self._build_markdown_blocks(d["parsed_markdown"], target_chars=target_block_chars)
            for d in valid_docs
        ]

        # Short-circuit: entire corpus fits within the synthesis budget
        if total_chars <= budget:
            sources: list[dict] = []
            for d, blocks in zip(valid_docs, doc_blocks_list):
                fname = d.get("filename") or d.get("doc_id")
                subj = d.get("subject") or "General"
                for blk in blocks:
                    section_label = (
                        f"{blk['heading']} ({blk['pct_start']}%–{blk['pct_end']}%)"
                        if blk["heading"]
                        else f"Section ({blk['pct_start']}%–{blk['pct_end']}%)"
                    )
                    sources.append(
                        {
                            "doc_id": d.get("doc_id"),
                            "filename": fname,
                            "subject": subj,
                            "section": section_label,
                            "text": blk["text"],
                        }
                    )
            return sources, heading_outline

        # Stratified Uniform Sampling across all matching documents
        total_slots = max(len(valid_docs), max(1, budget // target_block_chars))
        per_block_cap = min(target_block_chars, max(800, budget // total_slots))

        doc_lengths = [len(d["parsed_markdown"].strip()) for d in valid_docs]
        doc_block_counts = [len(b) for b in doc_blocks_list]
        slot_allocations = self._allocate_block_slots(
            doc_lengths=doc_lengths,
            doc_block_counts=doc_block_counts,
            total_slots=total_slots,
        )

        sources = []
        for d, blocks, k in zip(valid_docs, doc_blocks_list, slot_allocations):
            fname = d.get("filename") or d.get("doc_id")
            subj = d.get("subject") or "General"
            sampled_blocks = self._sample_doc_blocks(
                blocks=blocks,
                k=k,
                max_block_chars=per_block_cap,
            )
            for blk in sampled_blocks:
                section_label = (
                    f"{blk['heading']} ({blk['pct_start']}%–{blk['pct_end']}%)"
                    if blk["heading"]
                    else f"Sampled Segment ({blk['pct_start']}%–{blk['pct_end']}%)"
                )
                sources.append(
                    {
                        "doc_id": d.get("doc_id"),
                        "filename": fname,
                        "subject": subj,
                        "section": section_label,
                        "text": blk["text"],
                    }
                )

        return sources, heading_outline

    def get_synthesis_sources(
        self,
        user_id: str,
        subject: Optional[str] = None,
        doc_id: Optional[str] = None,
        max_total_chars: Optional[int] = None,
    ) -> list[dict]:
        """Backward-compatible helper returning only the stratified source blocks."""
        sources, _ = self.get_synthesis_bundle(
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
            max_total_chars=max_total_chars,
        )
        return sources

    def get_synthesis_context(
        self,
        user_id: str,
        subject: Optional[str] = None,
        doc_id: Optional[str] = None,
        max_total_chars: Optional[int] = None,
    ) -> str:
        """Backward-compatible helper returning a plain formatted Markdown synthesis context."""
        sources, outline = self.get_synthesis_bundle(
            user_id=user_id,
            subject=subject,
            doc_id=doc_id,
            max_total_chars=max_total_chars,
        )
        if not sources:
            return ""

        parts: list[str] = []
        if outline:
            parts.append(f"### Document Outline\n{outline}")
        for s in sources:
            parts.append(
                f"### Document: {s['filename']} — {s['section']} (Subject: {s['subject']})\n\n{s['text']}"
            )
        return "\n\n---\n\n".join(parts)
    
    def delete_last_interaction(self, chat_id: str):
        """Deletes the most recent user message and any subsequent assistant messages."""
        last_user_msg = self.messages.find_one(
            {"chat_id": chat_id, "role": "user"},
            sort=[("created_at", DESCENDING)]
        )
        if last_user_msg:
            self.messages.delete_many({
                "chat_id": chat_id,
                "created_at": {"$gte": last_user_msg["created_at"]}
            })