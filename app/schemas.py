# app/schemas.py
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel


# ---------- Sequential Ingestion Queue Job ----------
@dataclass
class IngestionJob:
    file_path: str
    clean_name: str
    content_type: str
    user_id: str
    subject: str
    doc_id: str


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