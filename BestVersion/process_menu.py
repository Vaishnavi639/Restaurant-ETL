#!/usr/bin/env python3
"""
MENU EXTRACTION PIPELINE (TEXT-ONLY MODE)

This script:
  1. Extracts text from PDFs/images
  2. Parses menu items using GPT (strict: items must have prices)
  3. Saves results to CSV
  4. Shows preview + summary
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

from restaurant_etl.extractors.universal_extractor import UniversalExtractor
from restaurant_etl.parsers.llm_parser import LLMMenuParser

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# PROCESS SINGLE FILE
# ============================================================

def process_single_menu(file_path: str, output_dir: str = "output"):
    print("\n" + "=" * 70)
    print("  MENU EXTRACTION PIPELINE (TEXT-ONLY MODE)")
    print("=" * 70 + "\n")

    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    restaurant_name = file_path.stem.replace("_", " ").title()

    print(f" File: {file_path.name}")
    print(f" Restaurant: {restaurant_name}")
    print(f" Mode: TEXT ONLY\n")

    # -----------------------------------------
    # STEP 1 — EXTRACT TEXT
    # -----------------------------------------
    print(" STEP 1: Extracting text…")
    print("-" * 70)

    extractor = UniversalExtractor()
    extraction = extractor.extract(str(file_path))

    if not extraction["success"]:
        print(" Extraction failed!")
        print(extraction.get("error", "Unknown error"))
        return None

    print(f" Extracted {extraction['char_count']} characters")
    print(f"   Method: {extraction['extraction_method']}\n")

    # -----------------------------------------
    # STEP 2 — PARSE WITH GPT (text-only)
    # -----------------------------------------
    print(" STEP 2: Parsing using GPT-4…")
    print("-" * 70)

    parser = LLMMenuParser()
    menu_data = parser.parse_menu(
        extraction["text"],
        restaurant_name=restaurant_name
    )

    print(f" Parsed {menu_data.total_items} items\n")

    # -----------------------------------------
    # STEP 3 — SAVE RESULTS
    # -----------------------------------------
    print(" STEP 3: Saving results…")
    print("-" * 70)

    df = menu_data.to_dataframe()
    if df.empty:
        print(" No items extracted!")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{file_path.stem}_extracted_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    print(f" Saved to: {out_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}\n")

    # -----------------------------------------
    # STEP 4 — PREVIEW
    # -----------------------------------------
    print(" PREVIEW (First 15 items)")
    print("=" * 70)

    preview_cols = ["item_name", "category", "price_display"]
    cols = [c for c in preview_cols if c in df.columns]
    print(df[cols].head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70 + "\n")

    return df


# ============================================================
# PROCESS FOLDER (BATCH MODE)
# ============================================================

def process_folder(input_folder: str = "input", output_folder: str = "output"):
    input_folder = Path(input_folder)

    if not input_folder.exists():
        print(f" Folder not found: {input_folder}")
        return

    supported = [".pdf", ".jpg", ".jpeg", ".png"]
    files = [f for f in input_folder.iterdir() if f.suffix.lower() in supported]

    if not files:
        print(f" No supported files in {input_folder}")
        return

    print(f"\n Found {len(files)} file(s) to process\n")

    results = []
    for idx, file in enumerate(files, 1):
        print("\n" + "=" * 70)
        print(f" Processing {idx}/{len(files)}: {file.name}")
        print("=" * 70 + "\n")

        try:
            df = process_single_menu(str(file), output_dir=output_folder)
            results.append({
                "file": file.name,
                "status": "success" if df is not None else "failed",
                "items": 0 if df is None else len(df)
            })
        except Exception as e:
            results.append({
                "file": file.name,
                "status": "error",
                "error": str(e)
            })

    # SUMMARY
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)

    for r in results:
        icon = "" if r["status"] == "success" else ""
        items = r.get("items", "-")
        print(f"{icon} {r['file']}: {items} items")


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Menu ETL (text-only mode)")

    parser.add_argument("input", nargs="?", default="input", help="File or folder")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--batch", action="store_true", help="Process all files in folder")

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        process_folder(str(input_path), args.output)
    elif input_path.is_file():
        process_single_menu(str(input_path), args.output)
    else:
        print(f" Path not found: {input_path}")


if __name__ == "__main__":
    main()
