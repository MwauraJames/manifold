# app/ingestion/parse.py
import os
from pathlib import Path
from langsmith import traceable

# Restrict PyTorch/ONNX threads BEFORE importing docling so it doesn't freeze a 2-core CPU
#os.environ["OMP_NUM_THREADS"] = "2"
#os.environ["MKL_NUM_THREADS"] = "2"

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.configs.config import Settings

settings = Settings()
PARSED_DIR = Path(settings.parsed_dir)
PARSED_DIR.mkdir(exist_ok=True)


def _build_converter() -> DocumentConverter:
    """
    Configure Docling specifically for a 2-core CPU handling CS & Math PDFs:
      - Disables full-page bitmap OCR (massive CPU saver for digital PDFs).
      - Enables TableStructure (FAST mode to save CPU while keeping Markdown tables).
      - Enables Code and Formula (LaTeX) enrichment.
      - Caps CPU threads at 2.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Set True ONLY if uploading scanned/photographed handwritten notes
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST  # Use ACCURATE if tables are complex & you can wait
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# Singleton converter so models are loaded into RAM once, not on every upload
_converter = None


def get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter


@traceable(name="parse_document")
def parse_document(file_path: str) -> str:
    """
    Locally parses a PDF into Markdown with LaTeX equations, code blocks, and tables.
    Drop-in replacement for LlamaParse.
    """
    converter = get_converter()
    result = converter.convert(file_path)

    # Export to the exact same Markdown format LlamaParse produced
    markdown_text = result.document.export_to_markdown()

    # Save a local copy in parsed_content/ for inspection, matching your config
    out_file = PARSED_DIR / f"{Path(file_path).stem}.md"
    out_file.write_text(markdown_text, encoding="utf-8")

    return markdown_text