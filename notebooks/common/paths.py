"""Centralized path constants for the data pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "embeddings_cache"
WOB_PATH = PROJECT_ROOT.parent / "words-of-brandon" / "wob_entries.json"
TAG_CLASS_PATH = DATA_DIR / "tag_classifications.json"
