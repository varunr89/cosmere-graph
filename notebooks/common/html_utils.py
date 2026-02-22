"""HTML text cleaning utilities."""

import re
from html import unescape


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from WoB entry text."""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()
