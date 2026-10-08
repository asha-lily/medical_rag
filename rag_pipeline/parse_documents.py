from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import TableItem
from langchain_core.documents import Document

_pipeline_options = PdfPipelineOptions()
# MPS (Apple Silicon) doesn't support float64, which the layout model requires.
_pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)


def parse_document(document_path: Path) -> list[Document]:
    """Parse a PDF into one `Document` per section, using Docling's layout-aware
    heading detection to find section boundaries."""
    medicine_name = (
        Path(document_path).stem.replace("_PIL", "").replace("_", " ").capitalize()
    )
    source = Path(document_path).name

    docling_doc = _converter.convert(str(document_path)).document

    sections = []
    current = None
    for item, _level in docling_doc.iterate_items():
        if isinstance(item, TableItem):
            text = item.export_to_markdown(docling_doc)
        else:
            text = getattr(item, "text", "")

        page = item.prov[0].page_no if item.prov else None
        if item.label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            current = {"heading": text, "text": "", "page": page}
            sections.append(current)
        else:
            if current is None:
                current = {"heading": None, "text": "", "page": page}
                sections.append(current)
            current["text"] += text + "\n"

    return [
        Document(
            page_content=section["text"].strip(),
            metadata={
                "medicine_name": medicine_name,
                "source": source,
                "heading": section["heading"],
                "page": section["page"],
                "section_index": section_index,
            },
        )
        for section_index, section in enumerate(sections)
        if section["text"].strip()
    ]
