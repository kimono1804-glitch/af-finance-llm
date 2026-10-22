"""Detect the reporting units of a table ("₹ in lakhs", "USD '000", "All figures in INR Cr").

Returns labels in the same form the training data uses ("INR lakh", "INR crore", "USD thousands",
"USD millions"), so the model sees one spelling for each unit. Only scaled units are detected here; a plain
currency ("Units: INR") is handled by the ingester's `Units:` row parsing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_CURRENCY = {"₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupees": "INR", "usd": "USD", "us$": "USD",
             "$": "USD", "eur": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP"}
_SCALE = {"lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh",
          "crore": "crore", "crores": "crore", "cr": "crore", "cr.": "crore", "crs": "crore",
          "thousand": "thousands", "thousands": "thousands", "'000": "thousands", "'000s": "thousands",
          "000": "thousands", "000s": "thousands", "k": "thousands",
          "million": "millions", "millions": "millions", "mn": "millions", "mm": "millions", "mio": "millions",
          "m": "millions"}

_CUR = r"(?P<cur>₹|rs\.?|inr|rupees|usd|us\$|\$|eur|€|gbp|£)"
_SC = r"(?P<scale>lakhs?|lacs?|crores?|crs?|cr\.?|thousands?|'000s?|000s?|millions?|mn|mm|mio|k|m)"
# currency then scale: "INR lakhs", "₹ in lakhs", "Rs. Lakh", "$000s", "USD '000", "US$ mn", "INR Cr"
_CUR_SCALE = re.compile(rf"(?<![a-z]){_CUR}\s*(?:in\s+)?{_SC}(?![a-z0-9])", re.IGNORECASE)
# scale then currency: "lakhs of rupees", "crore INR"
_SCALE_CUR = re.compile(rf"(?<![a-z]){_SC}\s*(?:of\s+)?{_CUR}(?![a-z])", re.IGNORECASE)
# "in lakhs", "figures in crores", "amounts in thousands": long words only (never bare k / m / cr)
_IN_SCALE = re.compile(r"\bin\s+(?P<scale>lakhs?|lacs?|crores?|thousands?|millions?)\b", re.IGNORECASE)


def _norm_quote(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def detect_units(text: str | None) -> str | None:
    """'(₹ in lakhs)' -> 'INR lakh'; 'USD 000s' -> 'USD thousands'; 'All figures in crores' -> 'INR crore'."""
    if not text:
        return None
    t = _norm_quote(text)
    for rx in (_CUR_SCALE, _SCALE_CUR):
        m = rx.search(t)
        if m:
            cur = _CURRENCY[m.group("cur").lower()]
            scale = _SCALE[m.group("scale").lower()]
            if scale in ("lakh", "crore") and cur != "INR":
                continue  # lakh / crore are Indian units; "$ cr" is almost certainly something else
            return f"{cur} {scale}"
    m = _IN_SCALE.search(t)
    if m:
        scale = _SCALE[m.group("scale").lower()]
        cur = next((_CURRENCY[c.lower()] for c in re.findall(r"₹|\binr\b|\busd\b|\$|\brupees\b", t, re.IGNORECASE)), None)
        if scale in ("lakh", "crore"):
            return f"INR {scale}"
        return f"{cur} {scale}" if cur else scale
    return None


def first_units(texts: Iterable[str | None]) -> str | None:
    """Units from the first text that states them (most specific source first)."""
    for t in texts:
        u = detect_units(t)
        if u:
            return u
    return None


# Plain currency without a scale: "Cash Summary (INR)", "(₹)", "amounts in USD", "Currency: INR".
_CUR_ONLY = re.compile(r"\(\s*(₹|inr|rs\.?|usd|us\$|\$|eur|€|gbp|£)\s*\)|\bin\s+(inr|usd|rupees|eur|gbp)\b"
                       r"|\bcurrency\s*[:=]\s*(inr|usd|eur|gbp)\b", re.IGNORECASE)


def detect_currency(text: str | None) -> str | None:
    """'Cash Summary (INR)' -> 'INR'. Used only when no scaled units were found."""
    if not text:
        return None
    m = _CUR_ONLY.search(text)
    if not m:
        return None
    return _CURRENCY[next(g for g in m.groups() if g).lower()]


def first_currency(texts: Iterable[str | None]) -> str | None:
    for t in texts:
        c = detect_currency(t)
        if c:
            return c
    return None
