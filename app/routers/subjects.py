# app/routers/subjects.py
from fastapi import APIRouter, Depends, HTTPException

from account.auth import get_current_user
from dependencies import logger, mongo, storage
from ingestion.reset_store import delete_by_subject
from schemas import SubjectCreate

router = APIRouter(prefix="/subjects", tags=["Subjects"])


@router.post("", status_code=201)
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


@router.get("")
def list_subjects(user_id: str = Depends(get_current_user)):
    subjects = mongo.list_user_subjects(user_id)
    logger.debug("[Subjects:List] user_id=%s | count=%d", user_id, len(subjects))
    return {"subjects": subjects}


@router.delete("/{subject:path}")
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