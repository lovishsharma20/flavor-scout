"""
Stream Amazon Reviews 2023 Sports_and_Outdoors item metadata.

Official source (historical, not live):
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz

Join key (official docs): parent_asin.
Only metadata for requested parent_asin values is kept in memory.
The full metadata file is not saved locally.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any, Iterable
from urllib.request import Request, urlopen

from config import AMAZON_META_URL

logger = logging.getLogger(__name__)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_as_text(v) for v in value]
        return " ".join(p for p in parts if p)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_brand(item: dict[str, Any]) -> str:
    details = item.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    if isinstance(details, dict):
        for key in ("Brand", "brand", "Manufacturer", "manufacturer"):
            brand = details.get(key)
            if brand:
                return str(brand).strip()
    store = item.get("store")
    if store:
        return str(store).strip()
    return ""


def _extract_categories(item: dict[str, Any]) -> str:
    cats = item.get("categories") or []
    if isinstance(cats, str):
        return cats.strip()
    if isinstance(cats, list):
        return " > ".join(str(c).strip() for c in cats if str(c).strip())
    return ""


def metadata_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": (item.get("parent_asin") or "").strip(),
        "product_title": (item.get("title") or "").strip(),
        "brand": _extract_brand(item),
        "store": (item.get("store") or "").strip() or None,
        "main_category": (item.get("main_category") or "").strip() or None,
        "product_categories": _extract_categories(item) or None,
        "product_description": _as_text(item.get("description")) or None,
        "average_rating": item.get("average_rating"),
        "rating_number": item.get("rating_number"),
        "price": item.get("price"),
    }


def fetch_metadata_for_asins(
    parent_asins: Iterable[str],
    url: str = AMAZON_META_URL,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Stream category metadata and keep rows whose parent_asin is in parent_asins.

    Stops early once every requested ID has been found.
    Missing IDs are left unmatched (no fabricated metadata).
    """
    wanted = {str(a).strip() for a in parent_asins if str(a).strip()}
    found: dict[str, dict[str, Any]] = {}
    stats = {"meta_scanned": 0, "meta_malformed": 0, "meta_matched": 0}

    if not wanted:
        return found, stats

    request = Request(
        url,
        headers={"User-Agent": "FlavorScout/0.1 (research MVP; metadata stream)"},
    )
    logger.info("Opening metadata stream for %s parent_asin values", len(wanted))
    with urlopen(request, timeout=300) as response:  # noqa: S310 - fixed official URL
        with gzip.GzipFile(fileobj=response) as gz:
            for raw_line in gz:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                stats["meta_scanned"] += 1
                if stats["meta_scanned"] % 100_000 == 0:
                    logger.info(
                        "Metadata scanned=%s matched=%s / wanted=%s",
                        stats["meta_scanned"],
                        len(found),
                        len(wanted),
                    )
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats["meta_malformed"] += 1
                    continue
                parent = (item.get("parent_asin") or "").strip()
                if parent not in wanted or parent in found:
                    continue
                found[parent] = metadata_row(item)
                if len(found) >= len(wanted):
                    logger.info(
                        "Matched all requested parent_asin values after scanning %s metadata rows.",
                        stats["meta_scanned"],
                    )
                    break

    stats["meta_matched"] = len(found)
    logger.info(
        "Metadata lookup finished: scanned=%s matched=%s missing=%s",
        stats["meta_scanned"],
        stats["meta_matched"],
        len(wanted) - stats["meta_matched"],
    )
    return found, stats
