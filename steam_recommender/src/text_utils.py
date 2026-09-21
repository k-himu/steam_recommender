"""
Shared text-cleaning utilities.
"""

from __future__ import annotations

import html
import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including \\r\\n\\t) into single spaces and trim."""
    if not isinstance(text, str):
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pass: HTML stripped, whitespace normalized, lowercased."""
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text.lower()


def clean_text_preserve_case(text: str) -> str:
    """Same as clean_text but keeps original case (useful for extracting
    proper nouns like 'Windows 10' or 'NVIDIA GeForce' before matching)."""
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text
