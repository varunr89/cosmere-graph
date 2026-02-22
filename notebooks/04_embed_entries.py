"""
Embed all 16,282 raw WoB entries with 5 embedding models and cache results to disk.

Models:
- Azure OpenAI text-embedding-3-large (3072-dim)
- Azure AI Cohere embed-v4
- Azure AI Mistral-embed
- Google Gemini embedding-001
- Voyage-4

Usage:
    python 04_embed_entries.py                    # all 5 models
    python 04_embed_entries.py --models voyage,gemini  # subset
"""

import argparse
import json
import os
import re
import time
from html import unescape
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────

project_root = Path(__file__).parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)

data_dir = project_root / "data"
cache_dir = data_dir / "embeddings_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

wob_path = project_root.parent / "words-of-brandon" / "wob_entries.json"

# ── CLI ───────────────────────────────────────────────────────────────────────

ALL_MODELS = ["azure_openai", "azure_cohere", "azure_mistral", "gemini", "voyage"]

parser = argparse.ArgumentParser(description="Embed WoB entries with multiple models")
parser.add_argument(
    "--models",
    type=str,
    default=",".join(ALL_MODELS),
    help="Comma-separated list of models to run (default: all 5)",
)
args = parser.parse_args()
requested_models = [m.strip() for m in args.models.split(",")]

for m in requested_models:
    if m not in ALL_MODELS:
        parser.error(f"Unknown model '{m}'. Choose from: {', '.join(ALL_MODELS)}")


# ── Text preparation ─────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    # Decode standard HTML entities
    text = unescape(text)
    return text.strip()


def prepare_entry_text(entry: dict) -> str:
    """Concatenate all lines and note into a single embedding-ready string."""
    parts = []
    for line in entry.get("lines", []):
        cleaned = strip_html(line.get("text", ""))
        if cleaned:
            parts.append(cleaned)
    note = entry.get("note", "")
    if note and note.strip():
        parts.append(strip_html(note))
    text = "\n".join(parts)
    # Truncate very long entries (rare edge cases)
    return text[:30_000]


print(f"Loading entries from {wob_path}...")
with open(wob_path) as f:
    raw_entries = json.load(f)

print(f"Total entries: {len(raw_entries)}")

entry_ids = [e["id"] for e in raw_entries]
texts = [prepare_entry_text(e) for e in raw_entries]

# Save ordered entry IDs for alignment with embedding matrices
ids_path = cache_dir / "entry_ids.json"
with open(ids_path, "w") as f:
    json.dump(entry_ids, f)
print(f"Saved {len(entry_ids)} entry IDs to {ids_path}")

# Quick stats
lengths = [len(t) for t in texts]
print(f"Text lengths: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")
empty_count = sum(1 for t in texts if not t.strip())
if empty_count:
    print(f"WARNING: {empty_count} entries have empty text after cleaning")


# ── Batching + retry helpers ──────────────────────────────────────────────────

def batched(items: list, batch_size: int):
    """Yield successive batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def retry_with_backoff(fn, retries=3, base_delay=2.0):
    """Call fn(), retrying up to `retries` times with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{retries} after error: {exc}")
            print(f"  Waiting {delay:.0f}s...")
            time.sleep(delay)


def check_env_vars(*keys) -> bool:
    """Return True if all env vars are set and non-empty."""
    for key in keys:
        val = os.environ.get(key, "")
        if not val:
            return False
    return True


# ── Embedding functions ───────────────────────────────────────────────────────

def embed_azure_openai(texts: list[str]) -> np.ndarray:
    """Embed with Azure OpenAI text-embedding-3-large (3072-dim)."""
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-06-01",
    )
    all_embeddings = []
    batch_size = 256
    batches = list(batched(texts, batch_size))
    for i, batch in enumerate(batches):
        print(f"  Embedding azure_openai: batch {i+1}/{len(batches)} ({len(all_embeddings)}/{len(texts)} texts)...")

        def call(b=batch):
            return client.embeddings.create(input=b, model="text-embedding-3-large")

        response = retry_with_backoff(call)
        all_embeddings.extend([d.embedding for d in response.data])
        if i < len(batches) - 1:
            time.sleep(0.5)

    return np.array(all_embeddings, dtype=np.float32)


def embed_azure_cohere(texts: list[str]) -> np.ndarray:
    """Embed with Azure AI Cohere embed-v3-english via OpenAI-compatible endpoint."""
    from openai import AzureOpenAI

    # Cohere on Azure uses the OpenAI-compatible API format
    endpoint = os.environ["AZURE_AI_COHERE_ENDPOINT"].rstrip("/")
    # Strip /openai/v1 suffix if present -- AzureOpenAI client adds it
    for suffix in ["/openai/v1", "/openai"]:
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=os.environ["AZURE_AI_COHERE_API_KEY"],
        api_version="2024-06-01",
    )
    all_embeddings = []
    batch_size = 96
    batches = list(batched(texts, batch_size))
    for i, batch in enumerate(batches):
        print(f"  Embedding azure_cohere: batch {i+1}/{len(batches)} ({len(all_embeddings)}/{len(texts)} texts)...")

        def call(b=batch):
            return client.embeddings.create(input=b, model="Cohere-embed-v3-english")

        response = retry_with_backoff(call)
        all_embeddings.extend([d.embedding for d in response.data])
        if i < len(batches) - 1:
            time.sleep(0.5)

    return np.array(all_embeddings, dtype=np.float32)


def embed_azure_mistral(texts: list[str]) -> np.ndarray:
    """Embed with Azure AI Mistral-embed."""
    from azure.ai.inference import EmbeddingsClient
    from azure.core.credentials import AzureKeyCredential

    client = EmbeddingsClient(
        endpoint=os.environ["AZURE_AI_MISTRAL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_AI_MISTRAL_API_KEY"]),
    )
    all_embeddings = []
    batch_size = 96
    batches = list(batched(texts, batch_size))
    for i, batch in enumerate(batches):
        print(f"  Embedding azure_mistral: batch {i+1}/{len(batches)} ({len(all_embeddings)}/{len(texts)} texts)...")

        def call(b=batch):
            return client.embed(input=b)

        response = retry_with_backoff(call)
        all_embeddings.extend([item.embedding for item in response.data])
        if i < len(batches) - 1:
            time.sleep(0.5)

    return np.array(all_embeddings, dtype=np.float32)


def embed_gemini(texts: list[str]) -> np.ndarray:
    """Embed with Google Gemini embedding-001."""
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    all_embeddings = []
    batch_size = 100
    batches = list(batched(texts, batch_size))
    for i, batch in enumerate(batches):
        print(f"  Embedding gemini: batch {i+1}/{len(batches)} ({len(all_embeddings)}/{len(texts)} texts)...")

        def call(b=batch):
            return client.models.embed_content(model="gemini-embedding-001", contents=b)

        response = retry_with_backoff(call)
        all_embeddings.extend([emb.values for emb in response.embeddings])
        if i < len(batches) - 1:
            time.sleep(0.5)

    return np.array(all_embeddings, dtype=np.float32)


def embed_voyage(texts: list[str]) -> np.ndarray:
    """Embed with Voyage-4."""
    import voyageai

    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    all_embeddings = []
    batch_size = 128
    batches = list(batched(texts, batch_size))
    for i, batch in enumerate(batches):
        print(f"  Embedding voyage: batch {i+1}/{len(batches)} ({len(all_embeddings)}/{len(texts)} texts)...")

        def call(b=batch):
            return client.embed(texts=b, model="voyage-4")

        response = retry_with_backoff(call)
        all_embeddings.extend(response.embeddings)
        if i < len(batches) - 1:
            time.sleep(0.5)

    return np.array(all_embeddings, dtype=np.float32)


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "azure_openai": {
        "fn": embed_azure_openai,
        "env_keys": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"],
    },
    "azure_cohere": {
        "fn": embed_azure_cohere,
        "env_keys": ["AZURE_AI_COHERE_ENDPOINT", "AZURE_AI_COHERE_API_KEY"],
    },
    "azure_mistral": {
        "fn": embed_azure_mistral,
        "env_keys": ["AZURE_AI_MISTRAL_ENDPOINT", "AZURE_AI_MISTRAL_API_KEY"],
    },
    "gemini": {
        "fn": embed_gemini,
        "env_keys": ["GOOGLE_API_KEY"],
    },
    "voyage": {
        "fn": embed_voyage,
        "env_keys": ["VOYAGE_API_KEY"],
    },
}

# ── Main loop ─────────────────────────────────────────────────────────────────

results = {}

for model_name in requested_models:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    npy_path = cache_dir / f"{model_name}.npy"

    # Check cache
    if npy_path.exists():
        arr = np.load(npy_path)
        print(f"Skipping {model_name} (cached) -- shape {arr.shape}, {npy_path.stat().st_size / 1e6:.1f} MB")
        results[model_name] = {"status": "cached", "shape": arr.shape, "path": npy_path}
        continue

    # Check env vars
    info = MODEL_REGISTRY[model_name]
    if not check_env_vars(*info["env_keys"]):
        missing = [k for k in info["env_keys"] if not os.environ.get(k)]
        print(f"WARNING: Skipping {model_name} -- missing env vars: {', '.join(missing)}")
        results[model_name] = {"status": "skipped_no_key"}
        continue

    # Embed
    try:
        t0 = time.time()
        embeddings = info["fn"](texts)
        elapsed = time.time() - t0
        np.save(npy_path, embeddings)
        print(f"Saved {model_name}: shape {embeddings.shape}, {npy_path.stat().st_size / 1e6:.1f} MB ({elapsed:.1f}s)")
        results[model_name] = {"status": "embedded", "shape": embeddings.shape, "path": npy_path, "time": elapsed}
    except Exception as exc:
        print(f"ERROR: {model_name} failed after retries: {exc}")
        results[model_name] = {"status": "failed", "error": str(exc)}

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for model_name, info in results.items():
    status = info["status"]
    if status in ("cached", "embedded"):
        shape = info["shape"]
        size_mb = info["path"].stat().st_size / 1e6
        extra = f" ({info['time']:.1f}s)" if "time" in info else ""
        print(f"  {model_name:20s} {status:10s} shape={shape}  {size_mb:.1f} MB{extra}")
    elif status == "skipped_no_key":
        print(f"  {model_name:20s} SKIPPED (missing API key)")
    elif status == "failed":
        print(f"  {model_name:20s} FAILED: {info['error']}")
print()
