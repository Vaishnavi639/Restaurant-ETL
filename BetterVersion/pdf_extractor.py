import pdfplumber
from pathlib import Path
import logging
from typing import Dict
import os
import tempfile

from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from pdf2image import convert_from_path

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PDFExtractor:
    supported_formats = [".pdf"]

    def __init__(self):
        self._ocr_client = None

    # -------------------- PUBLIC API --------------------

    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        pdf_path = Path(pdf_path)

        logger.info(f"Starting extraction from: {pdf_path.name}")

        text_blocks = []

        # ---------- STEP 1: Normal text extraction ----------
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")
                for i, page in enumerate(pdf.pages, 1):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        text_blocks.append(f"--- Page {i} ---\n{txt.strip()}")
                        logger.info(f"✓ Page {i}: {len(txt)} characters")
                    else:
                        logger.warning(f"Page {i}: No text found")
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")

        combined = "\n\n".join(text_blocks).strip()

        # ---------- STEP 2: OCR fallback ----------
        if len(combined) < 100:
            logger.warning("Low-text PDF detected; using Azure OCR (page-by-page)")
            combined = self._azure_ocr_per_page(pdf_path)
            method = "azure_ocr"
        else:
            method = "text"

        return {
            "text": combined,
            "source_file": pdf_path.name,
            "extraction_method": method,
            "char_count": len(combined),
            "success": len(combined) > 0,
        }

    # -------------------- AZURE OCR --------------------

    def _get_ocr_client(self):
        if not self._ocr_client:
            endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
            key = os.getenv("AZURE_DOC_INTEL_KEY")

            if not endpoint or not key:
                raise ValueError("Azure Document Intelligence credentials missing")

            self._ocr_client = DocumentAnalysisClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key),
            )

            logger.info("✓ Azure OCR client initialized")

        return self._ocr_client

    def _azure_ocr_per_page(self, pdf_path: Path) -> str:
        client = self._get_ocr_client()
        pages_text = []

        images = convert_from_path(pdf_path, dpi=300)

        for idx, img in enumerate(images, 1):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                img.save(tmp.name, format="PNG")

                try:
                    with open(tmp.name, "rb") as f:
                        poller = client.begin_analyze_document(
                            model_id="prebuilt-read",
                            document=f,
                        )

                    result = poller.result()

                    lines = []
                    for page in result.pages:
                        for line in page.lines:
                            lines.append(line.content)

                    page_text = "\n".join(lines).strip()
                    if page_text:
                        pages_text.append(f"--- Page {idx} ---\n{page_text}")
                        logger.info(f"✓ OCR page {idx}: {len(lines)} lines")
                    else:
                        logger.warning(f"OCR page {idx}: no text")

                except Exception as e:
                    logger.error(f"OCR failed on page {idx}: {e}")

        return "\n\n".join(pages_text)
