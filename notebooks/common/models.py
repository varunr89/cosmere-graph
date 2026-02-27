"""Shared constants for the data pipeline."""

ALL_MODELS = ["azure_openai", "azure_cohere", "azure_mistral", "gemini", "voyage"]

MODEL_DISPLAY_NAMES = {
    "azure_openai": "Azure OpenAI",
    "azure_cohere": "Azure Cohere",
    "azure_mistral": "Azure Mistral",
    "gemini": "Gemini",
    "voyage": "Voyage",
}

EXCLUDE_TYPES = {"meta", "book"}

MIN_EDGE_WEIGHT = 2

DEFAULT_THRESHOLD = 0.5

DEFAULT_FLOOR = 0.60
