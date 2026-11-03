from langsmith import traceable
from pathlib import Path

from configs.config import Settings

settings = Settings()


@traceable
def parse_document(client, file_path):
    # Upload and parse a document
    file = client.files.create(file=file_path, purpose="parse")
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        processing_options={"cost_optimizer": {"enable": True}},
        expand=["markdown"],
    )

    # BUG FIX: this used to return only pages[0], so everything after page 1
    # was never ingested. Join every page.
    pages = [p.markdown for p in result.markdown.pages if getattr(p, "markdown", None)]
    if not pages:
        raise ValueError("Parsing returned no text.")
    
    text = "\n\n".join(pages)
    
    return text