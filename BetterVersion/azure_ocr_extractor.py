import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()


class AzureOCRExtractor:
    def __init__(self):
        endpoint = os.getenv("AZURE_DOC_INTEL_ENDPOINT")
        key = os.getenv("AZURE_DOC_INTEL_KEY")

        if not endpoint or not key:
            raise ValueError("Azure Document Intelligence credentials missing")

        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract readable text from scanned or image-based PDFs.
        """
        with open(pdf_path, "rb") as f:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read",
                document=f
            )

        result = poller.result()

        lines = []
        for page in result.pages:
            for line in page.lines:
                lines.append(line.content)

        return "\n".join(lines)
