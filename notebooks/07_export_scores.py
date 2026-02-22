"""
Export precomputed per-entity similarity scores for the interactive frontend.

For each entity tag (excluding meta/book types) with >= 3 explicitly tagged
entries, this script computes:
  - Multi-prototype reference embeddings (1, 2, or 3 via k-means)
  - Per-entry cosine similarity scores against each prototype
  - Calibration stats (mean, std, p10-p50 in steps of 5) from explicitly-tagged entries
  - Specificity score: log(total_entries / entries_above_floor)

Inputs:
- data/embeddings_cache/<model>.npy  -- entry embeddings (16282, dims)
- data/embeddings_cache/entry_ids.json -- ordered entry IDs matching rows
- data/tag_classifications.json -- tag name -> {type, count}
- ../../words-of-brandon/wob_entries.json -- raw WoB entries (for explicit tags)

Outputs:
- data/scores.json -- per-entity scores, calibration, and metadata

Usage:
    python 07_export_scores.py
    python 07_export_scores.py --model azure_openai --floor 0.70
    python 07_export_scores.py --two-proto-min 5 --three-proto-min 10
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

# -- Paths -------------------------------------------------------------------

project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
cache_dir = data_dir / "embeddings_cache"
wob_path = project_root.parent / "words-of-brandon" / "wob_entries.json"

# -- CLI ---------------------------------------------------------------------

ALL_MODELS = ["azure_openai", "azure_cohere", "azure_mistral", "gemini", "voyage"]

parser = argparse.ArgumentParser(
    description="Export per-entity similarity scores for the interactive frontend",
)
parser.add_argument(
    "--model",
    choices=ALL_MODELS,
    default="azure_openai",
    help="Embedding model to use (default: azure_openai)",
)
parser.add_argument(
    "--floor",
    type=float,
    default=0.60,
    help="Minimum max-across-prototypes score to include an entry (default: 0.60)",
)
parser.add_argument(
    "--two-proto-min",
    type=int,
    default=5,
    help="Minimum explicit entries for 2 prototypes (default: 5)",
)
parser.add_argument(
    "--three-proto-min",
    type=int,
    default=10,
    help="Minimum explicit entries for 3 prototypes (default: 10)",
)
args = parser.parse_args()

MIN_ENTRIES_FOR_REF = 3

# -- 1. Load cached embeddings and entry IDs ---------------------------------

npy_path = cache_dir / f"{args.model}.npy"
ids_path = cache_dir / "entry_ids.json"

if not npy_path.exists():
    raise FileNotFoundError(
        f"Embeddings not found at {npy_path}. "
        f"Run 04_embed_entries.py --models {args.model} first."
    )
if not ids_path.exists():
    raise FileNotFoundError(
        f"Entry IDs not found at {ids_path}. "
        f"Run 04_embed_entries.py first."
    )

embeddings = np.load(npy_path)
with open(ids_path) as f:
    entry_ids = json.load(f)

assert embeddings.shape[0] == len(entry_ids), (
    f"Shape mismatch: embeddings {embeddings.shape[0]} vs entry_ids {len(entry_ids)}"
)

eid_to_idx = {eid: idx for idx, eid in enumerate(entry_ids)}
total_entries = len(entry_ids)

print(f"Model: {args.model}")
print(f"Embeddings: {embeddings.shape} ({embeddings.dtype})")
print(f"Floor: {args.floor}")
print(f"Proto thresholds: two >= {args.two_proto_min}, three >= {args.three_proto_min}")

# -- 2. Load raw WoB entries and tag classifications -------------------------

with open(wob_path) as f:
    raw_entries = json.load(f)

with open(data_dir / "tag_classifications.json") as f:
    tag_class = json.load(f)

# Entity tags: everything except meta and book
EXCLUDE_TYPES = {"meta", "book"}
entity_tags = {t for t, info in tag_class.items() if info["type"] not in EXCLUDE_TYPES}

print(f"\nTotal raw entries: {len(raw_entries)}")
print(f"Entity tags (excl meta/book): {len(entity_tags)}")

# Build explicit entity tags per entry
entry_explicit_tags = {}
for e in raw_entries:
    eid = e["id"]
    explicit = [t for t in e["tags"] if t in entity_tags]
    if explicit:
        entry_explicit_tags[eid] = set(explicit)

print(f"Entries with explicit entity tags: {len(entry_explicit_tags)}")

# -- 3. L2-normalize all entry embeddings -----------------------------------

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
zero_mask = (norms.squeeze() == 0)
zero_count = int(zero_mask.sum())
if zero_count:
    print(f"  WARNING: {zero_count} entries have zero-norm embeddings")
norms = np.where(norms == 0, 1.0, norms)
entries_norm = (embeddings / norms).astype(np.float32)
entries_norm[zero_mask] = 0.0

# -- 4. Process each entity -------------------------------------------------

entities_output = {}
skipped = []
proto_distribution = {1: 0, 2: 0, 3: 0}

for tag in sorted(entity_tags):
    # Find all entries that have this tag explicitly and are in our embedding set
    tag_eids = [
        eid for eid, tags in entry_explicit_tags.items()
        if tag in tags and eid in eid_to_idx
    ]

    if len(tag_eids) < MIN_ENTRIES_FOR_REF:
        skipped.append((tag, len(tag_eids)))
        continue

    n_explicit = len(tag_eids)
    indices = [eid_to_idx[eid] for eid in tag_eids]
    tag_embeddings = entries_norm[indices]  # already L2-normalized

    # 4a. Determine prototype count
    if n_explicit < args.two_proto_min:
        n_proto = 1
    elif n_explicit < args.three_proto_min:
        n_proto = 2
    else:
        n_proto = 3

    proto_distribution[n_proto] += 1

    # 4b/4c. Compute prototype(s)
    if n_proto == 1:
        # Average embeddings, L2-normalize
        proto = tag_embeddings.mean(axis=0)
        pnorm = np.linalg.norm(proto)
        if pnorm > 0:
            proto = proto / pnorm
        proto_matrix = proto.reshape(1, -1)
    else:
        # K-means clustering
        kmeans = KMeans(
            n_clusters=n_proto,
            n_init=10,
            random_state=42,
        )
        kmeans.fit(tag_embeddings)
        centers = kmeans.cluster_centers_

        # L2-normalize each center
        center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
        center_norms = np.where(center_norms == 0, 1.0, center_norms)
        proto_matrix = (centers / center_norms).astype(np.float32)

    # 4d. Compute similarity: entries_norm @ proto_matrix.T -> (n_entries, n_proto)
    sim = entries_norm @ proto_matrix.T  # (n_entries, n_proto)

    # 4e. Calibration stats: for explicitly-tagged entries
    explicit_indices = np.array(indices)
    explicit_sims = sim[explicit_indices]  # (n_explicit, n_proto)
    explicit_max = explicit_sims.max(axis=1)  # max across prototypes

    cal_mean = float(np.mean(explicit_max))
    cal_std = float(np.std(explicit_max))
    calibration = {"mean": round(cal_mean, 4), "std": round(cal_std, 4)}
    for pct in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        key = f"p{pct}"
        calibration[key] = round(float(np.percentile(explicit_max, pct)), 4)

    # 4f. Specificity: log(total_entries / entries_above_floor)
    max_scores = sim.max(axis=1)  # (n_entries,)
    above_floor_mask = max_scores > args.floor
    entries_above_floor = int(above_floor_mask.sum())

    if entries_above_floor == 0:
        # Avoid division by zero -- very niche entity
        specificity = float(math.log(total_entries))
    else:
        specificity = float(math.log(total_entries / entries_above_floor))

    # 4g. Store all entries above floor
    scores_dict = {}
    for idx in np.where(above_floor_mask)[0]:
        eid = entry_ids[idx]
        score_arr = [round(float(s), 2) for s in sim[idx]]
        scores_dict[str(eid)] = score_arr

    entities_output[tag] = {
        "specificity": round(specificity, 4),
        "entries_above_floor": entries_above_floor,
        "calibration": calibration,
        "prototypes": n_proto,
        "scores": scores_dict,
    }

# -- 5. Build output --------------------------------------------------------

output = {
    "meta": {
        "model": args.model,
        "floor": args.floor,
        "total_entries": total_entries,
        "proto_thresholds": {
            "two": args.two_proto_min,
            "three": args.three_proto_min,
        },
    },
    "entities": entities_output,
}

# -- 6. Write scores.json ---------------------------------------------------

scores_path = data_dir / "scores.json"
with open(scores_path, "w") as f:
    json.dump(output, f, separators=(",", ":"))

# -- 7. Summary --------------------------------------------------------------

size_mb = scores_path.stat().st_size / (1024 * 1024)

print(f"\n{'='*60}")
print("EXPORT SUMMARY")
print(f"{'='*60}")
print(f"  Model: {args.model}")
print(f"  Floor: {args.floor}")
print(f"  Total entries: {total_entries}")
print(f"  Proto thresholds: two >= {args.two_proto_min}, three >= {args.three_proto_min}")
print(f"\n  Entities exported: {len(entities_output)}")
print(f"  Entities skipped (< {MIN_ENTRIES_FOR_REF} entries): {len(skipped)}")
print(f"\n  Prototype distribution:")
for n, count in sorted(proto_distribution.items()):
    print(f"    {n} prototype(s): {count} entities")
print(f"\n  scores.json: {size_mb:.1f} MB")

# Sample calibration
print(f"\n  Sample calibration:")
for sample in ["kaladin", "hoid", "cosmere", "kelsier"]:
    if sample in entities_output:
        ent = entities_output[sample]
        cal = ent["calibration"]
        print(
            f"    {sample}: mean={cal['mean']:.3f}, std={cal['std']:.3f}, "
            f"p50={cal['p50']:.3f}, proto={ent['prototypes']}, "
            f"spec={ent['specificity']:.2f}, above_floor={ent['entries_above_floor']}"
        )

# Sample specificity extremes
specs = [(name, ent["specificity"]) for name, ent in entities_output.items()]
specs.sort(key=lambda x: x[1])
print(f"\n  Lowest specificity (hub topics):")
for name, spec in specs[:5]:
    print(f"    {name}: {spec:.3f}")
print(f"\n  Highest specificity (niche topics):")
for name, spec in specs[-5:]:
    print(f"    {name}: {spec:.3f}")

print()
