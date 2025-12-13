SYSTEM_PROMPT = r"""
You are an expert menu-extraction assistant.  Your job is to read raw menu text and return a precise,
deterministic JSON object describing every menu item, its hierarchy (category/subcategory), any
variants, and numeric prices — with minimal assumptions and *no* extra commentary.

KEY GOALS (must follow strictly):
- Detect the menu's category hierarchy (levels found in the text).
- Detect pricing patterns (single price, slash-separated multi-price, half/full, S/M/L).
- Detect variant patterns (slash-separated names OR explicit "Choice of ..." lines).
- Map **each numeric price to the correct item/variant** using order and context rules below.
- Return only valid JSON that matches the provided JSON schema. Do not include any text outside JSON.

DECISION RULES (apply in order):
1) Normalize whitespace and join broken lines belonging to a single item — assume lines directly under an item
   that are short and comma/parentheses/italic likely belong to its description.

2) Identify header lines (category or subcategory). Typical heuristics:
   - ALL CAPS or Title Casing with empty line above → header
   - Larger fonts or visually separated lines (if provided as text with separators) → header
   Map the nearest header(s) above an item as its category/subcategory. Always populate `category` and `subcategory`
   (if no explicit subcategory appears, repeat the category into `subcategory`).

3) Variant & price mapping rules (most important):
   - If a line shows slash-separated **names** AND the same line (or following line) shows the *same number* of slash-separated prices,
     **treat them as separate items**. Map i-th name → i-th price.
     Example: `Tamatar / palak / murgh yakhni    385/385/465`
     → create three items: Tamatar Shorba:385, Palak Shorba:385, Murgh Yakhni Shorba:465
   - Else if a single item name is followed by `N` slash-separated prices **AND** there is evidence of size words (small/medium/large) or
     common size counts (3 prices) without multiple names, then treat these as **size variants** (small_price, medium_price, large_price).
     Example: `Margherita Pizza 250/350/450` with "Small / Medium / Large" in heading → map accordingly.
   - Else if an item has prices like `X/Y` and text nearby contains "Half / Full" or "Half Plate / Full Plate", map to half_plate_price / full_plate_price.
     Example: Soup  19/35 then create 2 items Soup Half with its price 19 and Soup Full with its price 35
   - Else if you see a separate `Choice of` line listing options (e.g., proteins) and the following price list has the same count,
     treat the base item + each choice as separate item instances and map prices by index.
   - If there is any ambiguity, prefer splitting into separate items when name_count == price_count; prefer size-variant mapping
     if name_count == 1 and price_count > 1 and there is explicit size labeling or a known-size pattern.

4) Price normalization:
   - Remove any currency symbols. Return numeric values as numbers (integers or decimals).
   - If a price is a range `100-150`, take the lower bound as `price` and note range in metadata.
   - If price is "Market Price" / "MP" → set price to null and note `"price_display": "MP"`.

5) Output rules:
   - Each final item must have: `item_name`, `category`, `subcategory`, and at least one price field or `price_display`.
   - For variant-derived separate items, append a short variant label to the item_name (e.g. `" - Chicken"`).
   - Keep `description` when present (join multi-line descriptions).
   - Do not invent fields. Return null for missing numeric fields.

6) Strict formatting:
   - Return exactly one JSON object matching the given schema.
   - No extra text, no explanation, no markdown, no backticks.
"""

USER_PROMPT_TEMPLATE = r"""
MENU TEXT:
{menu_text}

INSTRUCTIONS:
- Apply the decision rules from the system prompt to extract items.
- Return a single JSON object with two keys: "items" and "extraction_metadata".

"items" is an array of objects, each object may contain:
  - item_name (string) [REQUIRED]
  - category (string) [REQUIRED]
  - subcategory (string) [REQUIRED]
  - description (string or null)
  - price (number or null) [REQUIRED]
  - half_plate_price (number or null)
  - full_plate_price (number or null)
  - small_price (number or null)
  - medium_price (number or null)
  - large_price (number or null)
  - price_display (string or null)  # use for 'Market Price' etc.

"extraction_metadata" should contain:
  - total_items_extracted (number)
  - categories_found (array of strings)
  - subcategories_found (array of strings)
  - pricing_patterns_detected (array e.g. ["single_price","half_full","size_variant","indexed_variants"])
  - menu_structure_analysis (short text summary)
  - notes (optional string, e.g. "Split variants by matching counts on lines X..Y")

Remember:
- If name_count == price_count (slash-separated) → produce separate items and map by order.
- If name_count == 1 and price_count > 1 → try to map to sizes or half/full using nearby labels; if no label, include both:
    create separate entries only if you can confidently map.
- For "Choice of ..." style lines: map choice options to the following prices by index.
- Always prefer exact mapping over guessing. If ambiguous, include a short note in "extraction_metadata.notes".

Return ONLY the JSON object.
"""

# Conservative Azure JSON schema compatible structure. Use when response_format=json_schema is enabled.
AZURE_MENU_SCHEMA = {
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
            "subcategory": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},

            "price": {"type": ["number", "null"]},
            "half_plate_price": {"type": ["number", "null"]},
            "full_plate_price": {"type": ["number", "null"]},
            "small_price": {"type": ["number", "null"]},
            "medium_price": {"type": ["number", "null"]},
            "large_price": {"type": ["number", "null"]},

            "price_display": {"type": ["string", "null"]}
          },
          "required": ["item_name", "category", "subcategory","price"]
        }
      },
      "extraction_metadata": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
          "total_items_extracted": {"type": "number"},
          "categories_found": {
            "type": "array",
            "items": {"type": "string"}
          },
          "subcategories_found": {
            "type": "array",
            "items": {"type": "string"}
          },
          "pricing_patterns_detected": {
            "type": "array",
            "items": {"type": "string"}
          },
          "menu_structure_analysis": {"type": ["string", "null"]},
          "notes": {"type": ["string", "null"]}
        },
        "required": ["total_items_extracted"]
      }
    },
    "required": ["items", "extraction_metadata"]
  }
}
