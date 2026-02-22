"""Embedding loading, normalization, and entity reference computation."""

import json

import numpy as np

from .paths import CACHE_DIR


def load_embeddings(model_name):
    """
    Load cached embeddings and entry IDs for a model.

    Args:
        model_name: e.g. "azure_openai", "gemini"

    Returns:
        (embeddings, entry_ids) where embeddings is an ndarray and
        entry_ids is a list of string IDs aligned with rows.

    Raises:
        FileNotFoundError: if the .npy or entry_ids.json file is missing
    """
    npy_path = CACHE_DIR / f"{model_name}.npy"
    ids_path = CACHE_DIR / "entry_ids.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {npy_path}. "
            f"Run 04_embed_entries.py --models {model_name} first."
        )
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Entry IDs not found at {ids_path}. "
            f"Run 04_embed_entries.py first."
        )

    embeddings = np.load(npy_path)
    with open(ids_path) as f:
        entry_ids = json.load(f)

    if embeddings.shape[0] != len(entry_ids):
        raise ValueError(
            f"Shape mismatch: {npy_path.name} has {embeddings.shape[0]} rows "
            f"but entry_ids.json has {len(entry_ids)} entries"
        )

    return embeddings, entry_ids


def normalize_embeddings(embeddings):
    """
    L2-normalize embeddings, handling zero-norm rows safely.

    Args:
        embeddings: 2D ndarray (n_entries, dims)

    Returns:
        Normalized ndarray of same shape. Zero-norm rows become zero vectors.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_mask = (norms.squeeze() == 0)
    zero_count = int(zero_mask.sum())
    if zero_count:
        print(f"  WARNING: {zero_count} entries have zero-norm embeddings")
    norms = np.where(norms == 0, 1.0, norms)
    result = embeddings / norms
    result[zero_mask] = 0.0
    return result


def compute_entity_refs(embeddings, entity_tags, entry_explicit_tags, eid_to_idx,
                        min_entries=5):
    """
    Compute mean reference embeddings for each entity tag.

    For each entity, averages the embeddings of its explicitly-tagged entries
    and L2-normalizes the result.

    Args:
        embeddings: 2D ndarray (n_entries, dims)
        entity_tags: set of entity tag names
        entry_explicit_tags: dict mapping entry_id -> set/list of tag names
        eid_to_idx: dict mapping entry_id -> row index in embeddings
        min_entries: minimum tagged entries required to produce a reference

    Returns:
        dict mapping tag_name -> normalized reference embedding (1D ndarray)
    """
    refs = {}
    for tag in sorted(entity_tags):
        tag_eids = [eid for eid, tags in entry_explicit_tags.items()
                    if tag in tags and eid in eid_to_idx]
        if len(tag_eids) < min_entries:
            continue
        indices = [eid_to_idx[eid] for eid in tag_eids]
        tag_embeddings = embeddings[indices]
        ref = tag_embeddings.mean(axis=0)
        norm = np.linalg.norm(ref)
        if norm > 0:
            ref = ref / norm
        refs[tag] = ref
    return refs
