from dagster import asset
from pathlib import Path
import logging

from restaurant_etl.extractors.universal_extractor import UniversalExtractor
from restaurant_etl.parsers.llm_parser import LLMMenuParser

logger = logging.getLogger(__name__)


@asset
def menu_etl_asset() -> str:
    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "input"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Resolved project_root = {project_root}")
    logger.info(f"Resolved input_dir = {input_dir} exists={input_dir.exists()}")
    logger.info(f"Resolved output_dir = {output_dir} exists={output_dir.exists()}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    extractor = UniversalExtractor()
    parser = LLMMenuParser()

    results = []

    # iterate the input directory
    for file in sorted(input_dir.iterdir()):
        if file.suffix.lower() not in [".pdf", ".jpg", ".jpeg", ".png"]:
            logger.info(f"Skipping unsupported: {file.name}")
            continue

        logger.info(f"Processing file: {file.name}")

        extraction = extractor.extract(str(file))
        if not extraction.get("success"):
            logger.warning(f"Extraction failed for {file.name}")
            continue

        parsed = parser.parse_menu(extraction['text'], restaurant_name=file.stem)
        df = parsed.to_dataframe()

        csv_path = output_dir / f"{file.stem}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Wrote CSV: {csv_path}")

        results.append(str(csv_path))

    return f"Completed ETL. Generated {len(results)} file(s)."

