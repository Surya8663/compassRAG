"""
Semantic Chunking and Tagging Service.
Uses LangChain's RecursiveCharacterTextSplitter with natural sentence boundaries to split
redacted text into semantic chunks, attaching domain DocumentMetadata to each chunk.
"""

import logging
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.models.common import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class SemanticChunkerService:
    """
    Sentence-boundary-aware chunking service that splits redacted document pages into discrete
    semantic segments and attaches strict domain metadata.
    """

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.models.common import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class SemanticChunkerService:
    """
    Section-aware, sentence-boundary chunking service that keeps resume role headings,
    companies, dates, and related achievement bullets together.
    """

    KNOWN_SECTIONS = [
        "Summary",
        "Executive Summary",
        "Experience",
        "Work Experience",
        "Professional Experience",
        "Research Experience",
        "Research Assistant",
        "Education",
        "Skills",
        "Projects",
        "Certifications",
        "Publications",
    ]

    def __init__(
        self, chunk_size: int = 900, chunk_overlap: int = 150
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n– ", "\n- ", "\n• ", "\n", ". ", " "],
            keep_separator=True,
        )

    def _extract_chunk_metadata(self, text: str) -> dict[str, str | None]:
        """Extracts structural metadata from chunk text."""
        meta: dict[str, str | None] = {
            "section_name": None,
            "organization": None,
            "role": None,
            "date_range": None,
        }

        # Detect section name
        for sec in self.KNOWN_SECTIONS:
            if re.search(rf"\b{re.escape(sec)}\b", text, re.IGNORECASE):
                meta["section_name"] = sec
                break

        # Detect date range (e.g. "Nov 2025 – May 2026", "Mar 2024 - Jun 2024", "2023 – 2027")
        date_match = re.search(
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})[^\n\d]*\d{4}?\s*(?:–|-|to)\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})[^\n\d]*\d{0,4}))",
            text,
            re.IGNORECASE,
        )
        if date_match:
            meta["date_range"] = date_match.group(1).strip()

        # Detect common organization names in resume context
        org_match = re.search(
            r"\b(HiDevs|In-Biot|TJIT|T\.?\s*John Institute|Azure AI|Microsoft|Google|AWS|Oracle)\b",
            text,
            re.IGNORECASE,
        )
        if org_match:
            meta["organization"] = org_match.group(1).strip()

        # Detect role title
        role_match = re.search(
            r"\b(Generative AI Developer Intern|AI Intern|Research Collaborator|Research Assistant|Generative AI Engineer|Software Engineer)\b",
            text,
            re.IGNORECASE,
        )
        if role_match:
            meta["role"] = role_match.group(1).strip()

        return meta

    def _split_into_blocks(self, text: str) -> list[str]:
        """
        Splits raw text into natural section / role blocks before character splitting.
        Keeps role titles, company names, dates, and associated achievement bullets grouped.
        """
        lines = text.split("\n")
        blocks: list[str] = []
        current_block: list[str] = []

        # Isolate contact header lines (email, phone, github, linkedin)
        is_contact_line = lambda line: bool(
            re.search(r"(@|github\.com|linkedin\.com|\+\d{2})", line, re.IGNORECASE)
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                continue

            # If it's a contact detail header, keep it separate
            if is_contact_line(line_str) and not current_block:
                blocks.append(line_str)
                continue

            # Check if line looks like a main section header
            is_section_header = any(
                line_str.lower() == sec.lower() or line_str.lower().startswith(f"{sec.lower()}:")
                for sec in self.KNOWN_SECTIONS
            )

            if is_section_header and current_block:
                blocks.append("\n".join(current_block))
                current_block = [line_str]
            else:
                current_block.append(line_str)

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    def chunk_and_tag_page(
        self,
        redacted_text: str,
        document_id: str,
        source: str,
        page_number: int,
        ingestion_timestamp: datetime,
        tenant_id: str,
        version_id: str,
        ocr_confidence: float | None = None,
        base_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Splits `redacted_text` into semantic chunks preserving role & section boundaries,
        attaching strict `DocumentMetadata` to each segment.
        """
        if not redacted_text or not redacted_text.strip():
            logger.warning(
                "Received empty redacted text for document_id=%s, page_number=%d",
                document_id,
                page_number,
            )
            return []

        raw_blocks = self._split_into_blocks(redacted_text)
        segments: list[str] = []

        for block in raw_blocks:
            clean_block = block.strip()
            if not clean_block:
                continue
            # If block fits comfortably within chunk_size, keep intact
            if len(clean_block) <= self.chunk_size:
                segments.append(clean_block)
            else:
                # Use RecursiveCharacterTextSplitter for larger blocks
                sub_segs = self.splitter.split_text(clean_block)
                segments.extend([s.strip() for s in sub_segs if s.strip()])

        chunks: list[DocumentChunk] = []
        for idx, segment_text in enumerate(segments):
            clean_text = segment_text.strip()
            if not clean_text:
                continue

            meta_extracted = self._extract_chunk_metadata(clean_text)

            metadata = DocumentMetadata(
                source=source,
                page_number=page_number,
                ingestion_timestamp=ingestion_timestamp,
                ocr_confidence=ocr_confidence,
                tenant_id=tenant_id,
                version_id=version_id,
                section_name=meta_extracted["section_name"],
                organization=meta_extracted["organization"],
                role=meta_extracted["role"],
                date_range=meta_extracted["date_range"],
            )

            chunk_id = f"{document_id}_p{page_number}_c{base_index + idx}"
            chunk_obj = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=clean_text,
                chunk_index=base_index + idx,
                metadata=metadata,
            )
            chunks.append(chunk_obj)

        logger.debug(
            "Created %d semantic chunks for document_id=%s page_number=%d",
            len(chunks),
            document_id,
            page_number,
        )
        return chunks


# Singleton instance helper
_chunker_service_instance: SemanticChunkerService | None = None


def get_chunker_service() -> SemanticChunkerService:
    """
    Returns a cached singleton instance of SemanticChunkerService.
    """
    global _chunker_service_instance
    if _chunker_service_instance is None:
        _chunker_service_instance = SemanticChunkerService()
    return _chunker_service_instance
