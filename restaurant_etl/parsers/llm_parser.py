import os
import time
import json
import logging
from typing import List, Dict, Optional, Any

from jsonschema import validate, ValidationError
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from restaurant_etl.models.menu_models import MenuItem, MenuData

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def estimate_tokens(text: str):
    return max(1, int(len(text) / 4))

# -------------------------------
# STRICT JSON SCHEMA (Azure format)
# -------------------------------
MENU_JSON_SCHEMA = {
    "name": "menu_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "item_name": {"type": "string"},
                        "category": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},

                        "price": {"type": ["number", "null"]},
                        "half_plate_price": {"type": ["number", "null"]},
                        "full_plate_price": {"type": ["number", "null"]},
                        "small_price": {"type": ["number", "null"]},
                        "medium_price": {"type": ["number", "null"]},
                        "large_price": {"type": ["number", "null"]},

                        "price_display": {"type": ["string", "null"]}
                    },
                    "required": ["item_name"]
                }
            }
        },
        "required": ["items"]
    }
}


class LLMMenuParser:
    def __init__(self):
        from openai import AzureOpenAI 

        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not all([self.api_key, self.endpoint, self.deployment]):
            raise ValueError("Missing AZURE_OPENAI_* env vars")

        # Correct client for your SDK
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version="2024-08-01-preview"
        )

        self.max_tokens = 4096
        self.max_retries = 3
        self.request_timeout = 60

        logger.info(" AzureOpenAI client initialized (text-only mode).")

    def parse_menu(self, text: str, restaurant_name: Optional[str] = None):
        logger.info(" Starting menu parse...")
        from ..utils.clean_text import normalize_extracted_text
        clean = normalize_extracted_text(text)
        chunks = self._chunk_text(clean)
        logger.info(f"Menu split into {len(chunks)} chunk(s).")

        all_items = []

        for idx, chunk in enumerate(chunks, 1):
            logger.info(f" Chunk {idx}/{len(chunks)} (est tokens: {estimate_tokens(chunk)})")
            parsed = self._call_llm_with_retries(chunk)
            if parsed:
                all_items.extend(parsed.get("items", []))
        valid_items = []
        for item in all_items:
            try:
                obj = MenuItem(**item)
                if obj.has_any_price():
                    valid_items.append(obj)
            except Exception:
                continue

        md = MenuData(
            restaurant_name=restaurant_name or "Unknown",
            items=valid_items,
            total_items=len(valid_items),
            extraction_confidence=len(valid_items) / max(1, len(all_items))
        )

        logger.info(f"Parsed {md.total_items} items")
        return md
    def _call_llm_with_retries(self, chunk: str):
        backoff = 1
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_llm(chunk)
            except Exception as e:
                logger.error(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries:
                    logger.error(" LLM failed all retries.")
                    return None
                time.sleep(backoff)
                backoff *= 2
    def _call_llm(self, text_chunk: str) -> Dict:
        system_msg = {
            "role": "system",
            "content": "You extract structured menu JSON. Follow schema exactly."
        }

        user_msg = {
            "role": "user",
            "content": (
                "Extract menu items and numeric prices from the text below.\n"
                "Fill fields for:\n"
                "- price\n"
                "- half_plate_price\n"
                "- full_plate_price\n"
                "- small_price, medium_price, large_price\n"
                "Return null for any unavailable fields.\n\n"
                f"MENU TEXT:\n{text_chunk}"
            )
        }

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[system_msg, user_msg],
            temperature=0,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
            response_format={
                "type": "json_schema",
                "json_schema": MENU_JSON_SCHEMA
            }
        )

        raw = response.choices[0].message.content

        Path("logs").mkdir(exist_ok=True)
        with open("logs/last_llm_response.json", "w") as f:
            f.write(json.dumps({"raw": raw}, indent=2))
        data = json.loads(raw)
        validate(instance=data, schema=MENU_JSON_SCHEMA["schema"])

        return data
    def _chunk_text(self, text: str) -> List[str]:
        max_chars = 2000  # safe size for Azure structured JSON
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end

        return chunks



