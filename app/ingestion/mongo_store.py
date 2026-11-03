# app/ingestion/mongo_store.py
from datetime import datetime, timezone
from typing import Optional
from bson.objectid import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
import gridfs

from configs.config import Settings

settings = Settings()


class MongoStore:
    def __init__(self):
        self.client = MongoClient(settings.mongo_uri)
        self.db = self.client[settings.mongo_db_name]
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

    def add_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
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

    @staticmethod
    def _stratified_sample_markdown(text: str, budget_chars: int) -> str:
        """
        If a document's parsed_markdown exceeds `budget_chars`, splits it into
        logical blocks (~1,500 chars each) and selects evenly-spaced blocks
        from start to finish so the entire lecture/unit is represented.
        """
        text = (text or "").strip()
        if not text or len(text) <= budget_chars:
            return text

        # Split on markdown headings or double newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        blocks: list[str] = []
        current_block: list[str] = []
        current_len = 0
        target_block_size = 1500

        for p in paragraphs:
            if current_len + len(p) > target_block_size and current_block:
                blocks.append("\n\n".join(current_block))
                current_block = [p]
                current_len = len(p)
            else:
                current_block.append(p)
                current_len += len(p) + 2

        if current_block:
            blocks.append("\n\n".join(current_block))

        if len(blocks) <= 1:
            return text[:budget_chars]

        num_to_pick = max(1, budget_chars // target_block_size)
        if num_to_pick >= len(blocks):
            return "\n\n".join(blocks)[:budget_chars]

        # Evenly spaced indices across the whole document (start, middle, end)
        if num_to_pick == 1:
            indices = [0]
        else:
            indices = [
                round(i * (len(blocks) - 1) / (num_to_pick - 1))
                for i in range(num_to_pick)
            ]

        seen = set()
        sampled = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                sampled.append(blocks[idx])

        return "\n\n[...]\n\n".join(sampled)

    def get_synthesis_context(
        self,
        user_id: str,
        subject: Optional[str] = None,
        doc_id: Optional[str] = None,
        max_total_chars: Optional[int] = None,
    ) -> str:
        """
        Fetches `parsed_markdown` from ready documents for global synthesis queries
        (exam questions, flashcards, unit summaries) and applies stratified sampling
        across all matching documents to respect the LLM context window.
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
            return ""

        per_doc_budget = max(2500, budget // len(valid_docs))
        sections: list[str] = []

        for d in valid_docs:
            fname = d.get("filename") or d.get("doc_id")
            subj = d.get("subject") or "General"
            sampled_body = self._stratified_sample_markdown(
                d["parsed_markdown"], budget_chars=per_doc_budget
            )
            sections.append(
                f"### Document: {fname} (Subject: {subj})\n\n{sampled_body}"
            )

        return "\n\n---\n\n".join(sections)