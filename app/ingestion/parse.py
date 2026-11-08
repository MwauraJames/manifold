# app/ingestion/parse.py
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from langsmith import traceable

from configs.config import Settings
from generation.prompts import VISION_TRANSCRIPTION_SYSTEM_PROMPT
from generation.models import get_vision_model_candidates, invoke_llm_traced

logger = logging.getLogger(__name__)
settings = Settings()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

VISION_TRANSCRIPTION_SYSTEM_PROMPT = VISION_TRANSCRIPTION_SYSTEM_PROMPT

def is_image_file(file_path: str | Path, content_type: Optional[str] = None) -> bool:
    """Return True if the file extension or MIME type corresponds to a supported image."""
    if content_type and content_type.lower().strip() in IMAGE_MIME_TYPES:
        return True
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS


def _resolve_image_mime(file_path: str | Path, content_type: Optional[str] = None) -> str:
    """Normalize image MIME type for base64 data URI."""
    if content_type and content_type.lower().strip() in IMAGE_MIME_TYPES:
        mime = content_type.lower().strip()
        return "image/jpeg" if mime == "image/jpg" else mime

    guessed, _ = mimetypes.guess_type(str(file_path))
    if guessed in IMAGE_MIME_TYPES:
        return guessed

    ext = Path(file_path).suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mapping.get(ext, "image/png")


@traceable(name="parse_image_to_markdown")
def parse_image_to_markdown(
    file_path: str | Path,
    content_type: Optional[str] = None,
) -> str:
    """
    Encodes an image as base64 and transcribes it into structured Markdown
    (OCR, LaTeX, tables, and diagram descriptions) using a vision-capable LLM.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type = _resolve_image_mime(path, content_type)
    raw_bytes = path.read_bytes()
    if not raw_bytes:
        raise ValueError(f"Image file '{path.name}' is empty.")

    b64_image = base64.b64encode(raw_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_image}"

    messages = [
        {"role": "system", "content": VISION_TRANSCRIPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Transcribe and analyze this study image ('{path.name}') "
                        "into complete, structured Markdown."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ],
        },
    ]

    candidates = get_vision_model_candidates()
    last_error: Optional[Exception] = None

    for model_id in candidates:
        try:
            logger.info(
                "Transcribing image '%s' (%s) with vision model '%s'",
                path.name,
                mime_type,
                model_id,
            )
            markdown = invoke_llm_traced(
                model_name=model_id,
                messages=messages,
                timeout=90,
            )
            cleaned = (markdown or "").strip()
            if cleaned:
                return cleaned
            raise ValueError(f"Vision model '{model_id}' returned empty transcription.")
        except Exception as exc:
            logger.warning(
                "Vision transcription with '%s' failed for '%s': %s",
                model_id,
                path.name,
                exc,
            )
            last_error = exc

    raise RuntimeError(
        f"All vision model candidates ({candidates}) failed to transcribe '{path.name}'. "
        f"Last error: {last_error}"
    )


@traceable(name="parse_document")
def parse_document(
    client,
    file_path: str | Path,
    content_type: Optional[str] = None,
) -> str:
    """
    Unified ingestion parser:
      - Routes images (.png, .jpg, .jpeg, .webp) to Vision-to-Markdown via LiteLLM.
      - Routes documents (.pdf, .docx, .pptx, etc.) to LlamaParse.
    """
    if is_image_file(file_path, content_type=content_type):
        return parse_image_to_markdown(file_path=file_path, content_type=content_type)

    # Document path via LlamaCloud (LlamaParse)
    file = client.files.create(file=file_path, purpose="parse")
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        processing_options={"cost_optimizer": {"enable": True}},
        expand=["markdown"],
    )

    pages = [p.markdown for p in result.markdown.pages if getattr(p, "markdown", None)]
    if not pages:
        raise ValueError("Parsing returned no text.")

    return "\n\n".join(pages)