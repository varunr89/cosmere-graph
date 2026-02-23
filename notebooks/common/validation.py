"""Stratified k-fold splitting and data loading for validation framework."""

import json
from collections import Counter

import numpy as np

from .paths import DATA_DIR, WOB_PATH, TAG_CLASS_PATH
from .models import EXCLUDE_TYPES


def load_validation_data():
    """Load all data needed for validation.

    Returns:
        entry_explicit_tags: dict {eid: set of entity names}
        entity_tags: set of all entity tag names (excl meta/book)
        entry_ids: list of all entry IDs (matching embedding rows)
        eid_to_idx: dict {eid: row index in embeddings}
    """
    with open(TAG_CLASS_PATH) as f:
        tag_class = json.load(f)

    with open(WOB_PATH) as f:
        raw_entries = json.load(f)

    entity_tags = {t for t, info in tag_class.items() if info["type"] not in EXCLUDE_TYPES}

    entry_explicit_tags = {}
    for e in raw_entries:
        eid = e["id"]
        explicit = {t for t in e["tags"] if t in entity_tags}
        if explicit:
            entry_explicit_tags[eid] = explicit

    ids_path = DATA_DIR / "embeddings_cache" / "entry_ids.json"
    with open(ids_path) as f:
        entry_ids = json.load(f)
    eid_to_idx = {eid: idx for idx, eid in enumerate(entry_ids)}

    return entry_explicit_tags, entity_tags, entry_ids, eid_to_idx


def build_stratified_folds(entry_explicit_tags, n_folds=5, seed=42):
    """Build stratified k-fold splits for multi-label entity tags.

    Uses iterative stratification: assigns entries to folds one at a time,
    prioritizing the least-represented entity in each step to maintain
    proportional representation.

    Args:
        entry_explicit_tags: dict {eid: set of entity names}
        n_folds: number of folds
        seed: random seed

    Returns:
        list of dicts, each with:
            train: list of eids (3 folds)
            val: list of eids (1 fold)
            test: list of eids (1 fold)
    """
    rng = np.random.RandomState(seed)

    # Assign each entry to a fold using iterative stratification
    eids = list(entry_explicit_tags.keys())
    rng.shuffle(eids)

    # Count entity frequency
    entity_counts = Counter()
    for tags in entry_explicit_tags.values():
        for t in tags:
            entity_counts[t] += 1

    # Initialize fold assignments
    fold_assignment = {}
    fold_entity_counts = [Counter() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    # Sort entries by rarest entity first (hardest to stratify)
    def rarest_entity_freq(eid):
        tags = entry_explicit_tags[eid]
        return min(entity_counts[t] for t in tags)

    eids_sorted = sorted(eids, key=rarest_entity_freq)

    for eid in eids_sorted:
        tags = entry_explicit_tags[eid]
        # Pick the fold that most needs this entry's entities
        best_fold = None
        best_score = float("inf")
        for f in range(n_folds):
            # Score: how over-represented are this entry's entities in fold f?
            score = sum(
                fold_entity_counts[f][t] / max(1, entity_counts[t])
                for t in tags
            )
            # Break ties by fold size (prefer smaller folds)
            score += fold_sizes[f] * 0.001
            if score < best_score:
                best_score = score
                best_fold = f
        fold_assignment[eid] = best_fold
        for t in tags:
            fold_entity_counts[best_fold][t] += 1
        fold_sizes[best_fold] += 1

    # Build fold lists
    fold_eids = [[] for _ in range(n_folds)]
    for eid, f in fold_assignment.items():
        fold_eids[f].append(eid)

    # Create train/val/test splits (rotating which fold is val and test)
    folds = []
    for test_idx in range(n_folds):
        val_idx = (test_idx + 1) % n_folds
        train_indices = [i for i in range(n_folds) if i != test_idx and i != val_idx]
        train = []
        for i in train_indices:
            train.extend(fold_eids[i])
        folds.append({
            "train": train,
            "val": fold_eids[val_idx],
            "test": fold_eids[test_idx],
        })

    return folds
