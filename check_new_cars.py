#!/usr/bin/env python3
"""
check_new_cars.py — Independent detection layer for carandbike.com

Scrapes carandbike.com to discover new car listings, extracts basic metadata
(brand, model name, variants, url), and appends unseen cars into your
Excel tracking file (cars.xlsx) for automate.py to pick up.

Usage via CMD:
  # Check for new cars and update cars.xlsx
  python check_new_cars.py --excel cars.xlsx

  # Check without updating Excel (dry-run / preview only)
  python check_new_cars.py --excel cars.xlsx --preview

  # Automatically run full pipeline if new cars are found
  python check_new_cars.py --excel cars.xlsx --auto-run
"""

import argparse
import logging
import re
import sys
import subprocess
from pathlib import Path

try:
    from openpyxl import load_workbook, Workbook
except ImportError:
    print("ERROR: openpyxl is required. Run: pip install openpyxl", file=sys.stderr)
    sys.exit(1)

# Import helper functions from your existing scrapper.py
try:
    from scrapper import (
        fetch,
        _brand_model_from_url,
        _extract_embedded_variants,
        parse_sitemap,
    )
except ImportError:
    print("ERROR: scrapper.py must be in the same directory as check_new_cars.py", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CARANDBIKE_SITEMAP = "https://www.carandbike.com/sitemap.xml"
CARANDBIKE_BASE = "https://www.carandbike.com"

COLUMNS = [
    "brand",
    "model",
    "url",
    "scrape_status",
    "upload_status",
    "acf_json",
    "error",
    "scraped_at",
    "uploaded_at",
]


def load_existing_urls_and_models(excel_path: Path) -> tuple[set[str], set[str]]:
    """Loads existing URLs and Brand-Model pairs from cars.xlsx to avoid duplicates."""
    known_urls = set()
    known_models = set()

    if not excel_path.exists():
        log.warning("Excel file %s not found. A new one will be created.", excel_path)
        return known_urls, known_models

    wb = load_workbook(excel_path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    if not rows:
        return known_urls, known_models

    headers = [str(h).strip().lower() if h else "" for h in rows[0]]
    url_idx = headers.index("url") if "url" in headers else -1
    brand_idx = headers.index("brand") if "brand" in headers else -1
    model_idx = headers.index("model") if "model" in headers else -1

    for row in rows[1:]:
        url = str(row[url_idx]).strip().lower() if url_idx != -1 and row[url_idx] else ""
        brand = str(row[brand_idx]).strip().lower() if brand_idx != -1 and row[brand_idx] else ""
        model = str(row[model_idx]).strip().lower() if model_idx != -1 and row[model_idx] else ""

        if url:
            known_urls.add(url.rstrip("/"))
        if brand and model:
            known_models.add(f"{brand}|{model}")

    return known_urls, known_models


def fetch_car_details(url: str) -> dict:
    """Quickly extracts basic car details (brand, model, variants, URL) from the car page."""
    log.info("Fetching details for new car: %s", url)
    soup, html = fetch(url)
    
    brand, model = _brand_model_from_url(url)
    variants = []

    if html:
        # Extract embedded variants using existing scrapper.py logic
        raw_variants = _extract_embedded_variants(html)
        variants = [v.get("variant_name", "") for v in raw_variants if v.get("variant_name")]

    return {
        "brand": brand,
        "model": model,
        "url": url,
        "variants": variants,
        "variant_count": len(variants),
    }


def append_to_excel(excel_path: Path, new_cars: list[dict]) -> None:
    """Appends newly discovered cars to cars.xlsx."""
    if not excel_path.exists():
        wb = Workbook()
        ws = wb.active
        ws.title = "cars"
        ws.append(COLUMNS)
    else:
        wb = load_workbook(excel_path)
        ws = wb.active

    for car in new_cars:
        ws.append([
            car["brand"],
            car["model"],
            car["url"],
            "pending",
            "pending",
            "",
            "",
            "",
            ""
        ])

    wb.save(excel_path)
    log.info("Saved %d new car(s) to %s", len(new_cars), excel_path.resolve())


def check_for_new_cars(excel_path: Path, preview_only: bool = False) -> list[dict]:
    """Scrapes site links, filters out known ones, and extracts info for new products."""
    known_urls, known_models = load_existing_urls_and_models(excel_path)

    log.info("Scanning carandbike.com sitemap for car listings...")
    all_urls = parse_sitemap(CARANDBIKE_SITEMAP)

    if not all_urls:
        log.warning("No URLs retrieved from sitemap. Checking default structure...")
        return []

    # Find URLs not present in our Excel file
    new_urls = []
    for u in all_urls:
        clean_u = u.strip().rstrip("/")
        brand, model = _brand_model_from_url(clean_u)
        model_key = f"{brand.lower()}|{model.lower()}"

        if clean_u not in known_urls and model_key not in known_models:
            new_urls.append(clean_u)

    log.info("Found %d total cars in sitemap. Unprocessed/New cars: %d", len(all_urls), len(new_urls))

    new_car_details = []
    for i, url in enumerate(new_urls, 1):
        log.info("[%d/%d] New product detected!", i, len(new_urls))
        info = fetch_car_details(url)
        new_car_details.append(info)

        # Print summary in CMD
        print("\n" + "=" * 60)
        print(f" NEW PRODUCT FOUND ")
        print(f" Brand       : {info['brand']}")
        print(f" Model       : {info['model']}")
        print(f" URL         : {info['url']}")
        print(f" Variants    : {info['variant_count']} found -> {', '.join(info['variants'][:5])}{'...' if info['variant_count'] > 5 else ''}")
        print("=" * 60 + "\n")

    if new_car_details and not preview_only:
        append_to_excel(excel_path, new_car_details)

    return new_car_details


def main():
    parser = argparse.ArgumentParser(
        description="Detect newly uploaded cars on carandbike.com and register them in Excel."
    )
    parser.add_argument(
        "--excel",
        default="cars.xlsx",
        help="Path to Excel spreadsheet tracking cars (default: cars.xlsx)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only display new cars in console without updating Excel",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Automatically run automate.py pipeline if new cars are detected",
    )

    args = parser.parse_args()
    excel_path = Path(args.excel).resolve()

    new_cars = check_for_new_cars(excel_path, preview_only=args.preview)

    if not new_cars:
        log.info("No new cars found on site. System is up to date.")
        return 0

    log.info("Detection finished. Total new cars added: %d", len(new_cars))

    # Optional: trigger full pipeline automatically if requested
    if args.auto_run and not args.preview:
        log.info("Launching automate.py pipeline for new entries...")
        cmd = [sys.executable, "automate.py", str(excel_path), "--remove-logo", "--rephrase-content"]
        subprocess.run(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())