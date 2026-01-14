"""Utilities for parsing static rate tables such as kilometric rates."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from bs4 import BeautifulSoup  # type: ignore[import]

from app.core.logging import get_logger

logger = get_logger(__name__)

# Province and territory synonyms for lookup convenience
_PROVINCE_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "Alberta": ("alberta", "ab"),
    "British Columbia": ("british columbia", "bc"),
    "Manitoba": ("manitoba", "mb"),
    "New Brunswick": ("new brunswick", "nb"),
    "Newfoundland and Labrador": ("newfoundland", "newfoundland and labrador", "nl"),
    "Nova Scotia": ("nova scotia", "ns"),
    "Ontario": ("ontario", "on"),
    "Prince Edward Island": ("prince edward island", "pei", "pe"),
    "Quebec": ("quebec", "québec", "qc"),
    "Saskatchewan": ("saskatchewan", "sk"),
    "Yukon": ("yukon", "yt"),
    "Northwest Territories": ("northwest territories", "nwt", "nt"),
    "Nunavut": ("nunavut", "nu"),
    "Canada & USA": ("canada & usa", "canada and usa", "canada usa"),
    "Yukon & Alaska": ("yukon & alaska", "yukon alaska"),
}


def _data_root() -> Path:
    """Return the repository root directory."""
    utils_dir = Path(__file__).resolve().parent
    return utils_dir.parent.parent


@lru_cache(maxsize=1)
def _load_kilometric_table() -> Dict[str, Dict[str, str]]:
    """Parse the NJC kilometric rates table into a mapping."""
    root = _data_root().parent
    html_path = root / "source" / "njc" / "njc-travel-directive-v238.html"

    if not html_path.exists():
        logger.warning("Kilometric rate source file not found: %s", html_path)
        return {}

    try:
        html = html_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to read kilometric rate source file: %s", exc)
        return {}

    soup = BeautifulSoup(html, "html.parser")
    table = None
    for candidate in soup.find_all("table"):
        rows = candidate.find_all("tr")
        if not rows:
            continue

        first_row = rows[0]
        header_cells = [cell.get_text(strip=True).lower() for cell in first_row.find_all(["th", "td"])]
        if not header_cells:
            continue

        if any("province/territory" in cell for cell in header_cells) and any(
            "cents/km" in cell for cell in header_cells
        ):
            table = candidate
            break

    if table is None:
        logger.warning("Kilometric rate table not located in %s", html_path)
        return {}

    # Determine header positions
    header_row = [
        cell.get_text(strip=True).lower() for cell in table.find_all("tr")[0].find_all(["th", "td"])
    ]
    location_idx = None
    rate_idx = None
    for idx, value in enumerate(header_row):
        if "province/territory" in value:
            location_idx = idx
        if "cents/km" in value:
            rate_idx = idx

    if location_idx is None or rate_idx is None:
        logger.warning("Kilometric table headers missing expected columns: %s", header_row)
        return {}

    rates: Dict[str, Dict[str, str]] = {}
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) <= max(location_idx, rate_idx):
            continue
        location_raw = cells[location_idx]
        rate_raw = cells[rate_idx]
        if not location_raw or not rate_raw:
            continue

        rate_value = _parse_rate(rate_raw)
        if rate_value is None:
            continue

        canonical = _canonical_location_name(location_raw)
        rates[canonical] = {
            "raw_location": location_raw,
            "rate": f"{rate_value:.1f}",
        }

    effective_date = _extract_effective_date(table)
    for info in rates.values():
        info["effective_date"] = effective_date

    return rates


def _canonical_location_name(name: str) -> str:
    """Normalize the table location entry to a canonical key."""
    return re.sub(r"\s+", " ", name).strip()


def _parse_rate(value: str) -> Optional[float]:
    """Parse a rate string into a float."""
    cleaned = value.replace("¢", "").replace("cents", "").replace("per km", "")
    cleaned = cleaned.replace("taxes included", "").replace("%", "").strip()
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_effective_date(table) -> str:
    """Extract the effective date preceding the provided table."""
    node = table.find_previous(string=re.compile(r"Effective Date", re.IGNORECASE))
    if node:
        match = re.search(r"Effective Date:\s*([^\n<]+)", str(node))
        if match:
            return match.group(1).strip()
    return "Unknown"


def normalize_kilometric_location(text: Optional[str]) -> Optional[str]:
    """Return the canonical location name if the text references a known jurisdiction."""
    if not text:
        return None

    lower_text = text.lower()
    for canonical, synonyms in _PROVINCE_SYNONYMS.items():
        for synonym in synonyms:
            if re.search(rf"\b{re.escape(synonym)}\b", lower_text):
                return canonical
    return None


def get_kilometric_rate(location: Optional[str]) -> Optional[Tuple[str, float, str]]:
    """Return (canonical_name, rate_cents_per_km, effective_date) for the location."""
    if not location:
        return None

    canonical = normalize_kilometric_location(location)
    if not canonical:
        return None

    rates = _load_kilometric_table()
    if not rates:
        return None

    table_entry = None

    # Direct match on canonical name
    for key, info in rates.items():
        if normalize_kilometric_location(key) == canonical:
            table_entry = info
            break

    if not table_entry:
        return None

    try:
        rate_value = float(table_entry["rate"])
    except (ValueError, KeyError):
        return None

    effective_date = table_entry.get("effective_date", "July 1, 2025")
    raw_location = table_entry.get("raw_location", canonical)
    return raw_location, rate_value, effective_date
