"""
Build the Cosmere knowledge graph using embedding-based entity tagging.

Replaces regex text matching (03_text_matching.py) with cosine-similarity
tagging against averaged entity reference embeddings. For each entity, we
compute a reference embedding by averaging the embeddings of all entries
that carry that entity's explicit Arcanum tag, then tag every entry whose
embedding exceeds a cosine-similarity threshold with that entity.

Inputs:
- data/embeddings_cache/<model>.npy  -- entry embeddings (16282, dims)
- data/embeddings_cache/entry_ids.json -- ordered entry IDs matching rows
- data/tag_classifications.json -- tag name -> {type, count}
- ../../words-of-brandon/wob_entries.json -- raw WoB entries

Outputs:
- data/graph.json -- nodes + edges for the knowledge graph
- data/entries.json -- cleaned WoB entries with combined tags
- data/similarity.json -- top-K similar entities per node

Usage:
    python 05_build_graph.py
    python 05_build_graph.py --model gemini --threshold 0.6
    python 05_build_graph.py --model azure_openai --top-k 15
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# -- Paths -------------------------------------------------------------------

project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
cache_dir = data_dir / "embeddings_cache"
wob_path = project_root.parent / "words-of-brandon" / "wob_entries.json"

# -- CLI ---------------------------------------------------------------------

ALL_MODELS = ["azure_openai", "azure_cohere", "azure_mistral", "gemini", "voyage"]

parser = argparse.ArgumentParser(
    description="Build Cosmere knowledge graph with embedding-based tagging",
)
parser.add_argument(
    "--model",
    choices=ALL_MODELS,
    default="azure_openai",
    help="Embedding model to use (default: azure_openai)",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Cosine similarity threshold for tagging (default: 0.5)",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=10,
    help="Number of similar entities per node in similarity.json (default: 10)",
)
parser.add_argument(
    "--all-entities",
    action="store_true",
    help="Apply implicit tagging to ALL entities (default: only isolated nodes)",
)
args = parser.parse_args()


# -- Helpers -----------------------------------------------------------------

def strip_html(text):
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()


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

print(f"Model: {args.model}")
print(f"Embeddings: {embeddings.shape} ({embeddings.dtype})")
print(f"Threshold: {args.threshold}")
print(f"Top-K: {args.top_k}")

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

# Build lookup: eid -> raw entry
raw_by_id = {e["id"]: e for e in raw_entries}

# For each entry, extract explicit entity tags
entry_explicit_tags = {}
for e in raw_entries:
    eid = e["id"]
    explicit = [t for t in e["tags"] if t in entity_tags]
    if explicit:
        entry_explicit_tags[eid] = set(explicit)

print(f"Entries with explicit entity tags: {len(entry_explicit_tags)}")

# -- 3. Compute entity reference embeddings ---------------------------------

MIN_ENTRIES_FOR_REF = 5

entity_ids = []  # parallel to entity_refs rows
entity_ref_list = []
skipped_entities = []

for tag in sorted(entity_tags):
    # Find all entries that have this tag explicitly
    tag_eids = [
        eid for eid, tags in entry_explicit_tags.items()
        if tag in tags and eid in eid_to_idx
    ]

    if len(tag_eids) < MIN_ENTRIES_FOR_REF:
        skipped_entities.append((tag, len(tag_eids)))
        continue

    # Gather embedding rows
    indices = [eid_to_idx[eid] for eid in tag_eids]
    tag_embeddings = embeddings[indices]

    # Average and L2-normalize
    ref = tag_embeddings.mean(axis=0)
    norm = np.linalg.norm(ref)
    if norm > 0:
        ref = ref / norm

    entity_ids.append(tag)
    entity_ref_list.append(ref)

entity_refs = np.array(entity_ref_list, dtype=np.float32)

print(f"\nEntity reference embeddings: {entity_refs.shape}")
print(f"Entities with refs: {len(entity_ids)}")
print(f"Entities skipped (< {MIN_ENTRIES_FOR_REF} entries): {len(skipped_entities)}")

# Show a few skipped entities
if skipped_entities:
    skipped_sorted = sorted(skipped_entities, key=lambda x: -x[1])[:10]
    print("  Top skipped (by count):")
    for tag, count in skipped_sorted:
        print(f"    {tag}: {count} entries")

# -- 4. Find baseline isolated nodes ----------------------------------------

MIN_EDGE_WEIGHT = 2

# Build baseline co-occurrence graph from explicit tags only to find isolated nodes
baseline_edge_entries = defaultdict(list)
baseline_node_entries = defaultdict(list)

for eid, tags in entry_explicit_tags.items():
    tags_sorted = sorted(tags)
    for t in tags_sorted:
        baseline_node_entries[t].append(eid)
    for i in range(len(tags_sorted)):
        for j in range(i + 1, len(tags_sorted)):
            pair = (tags_sorted[i], tags_sorted[j])
            baseline_edge_entries[pair].append(eid)

baseline_connected = set()
for pair, eids in baseline_edge_entries.items():
    if len(eids) >= MIN_EDGE_WEIGHT:
        baseline_connected.add(pair[0])
        baseline_connected.add(pair[1])

baseline_isolated = {t for t in entity_tags if t not in baseline_connected}

# Determine which entities to apply implicit tagging to
entity_id_set = set(entity_ids)
if args.all_entities:
    target_entities = entity_id_set
    print(f"\nImplicit tagging mode: ALL entities ({len(target_entities)} with refs)")
else:
    target_entities = baseline_isolated & entity_id_set
    print(f"\nBaseline graph: {len(baseline_connected)} connected, {len(baseline_isolated)} isolated")
    print(f"Isolated with reference embeddings: {len(target_entities)}")
    print(f"Implicit tagging mode: ISOLATED ONLY ({len(target_entities)} entities)")

# -- 5. Embedding-based implicit tagging (targeted) -------------------------

# L2-normalize all entry embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
zero_mask = (norms.squeeze() == 0)
zero_count = int(zero_mask.sum())
if zero_count:
    print(f"  WARNING: {zero_count} entries have zero-norm embeddings (will be excluded)")
norms = np.where(norms == 0, 1.0, norms)
entries_norm = embeddings / norms
entries_norm[zero_mask] = 0.0

# Only compute similarity for target entity columns (not all 367)
target_cols = [i for i, eid in enumerate(entity_ids) if eid in target_entities]
target_names = [entity_ids[i] for i in target_cols]
target_refs = entity_refs[target_cols]

print(f"\nComputing similarity matrix ({entries_norm.shape[0]} x {len(target_cols)})...")
target_sim = entries_norm @ target_refs.T
target_sim = np.nan_to_num(target_sim, nan=0.0, posinf=0.0, neginf=0.0)

# Build combined tags for each entry
entry_all_tags = {}
implicit_tag_counts = Counter()
entries_enhanced = 0
total_implicit_additions = 0

for eid in entry_ids:
    explicit = entry_explicit_tags.get(eid, set())

    # Only add implicit tags for target entities, and only if the entry
    # already has other entity tags (so the implicit tag creates an edge)
    idx = eid_to_idx[eid]
    sims = target_sim[idx]
    predicted = set()
    for col, entity in enumerate(target_names):
        if sims[col] > args.threshold and entity not in explicit:
            # Only keep if entry has at least one other tag to co-occur with
            other_tags = explicit - {entity}
            if other_tags:
                predicted.add(entity)
                implicit_tag_counts[entity] += 1

    combined = explicit | predicted
    if combined:
        entry_all_tags[eid] = combined
        if predicted:
            entries_enhanced += 1
            total_implicit_additions += len(predicted)

print(f"\nEntries with any entity tag: {len(entry_all_tags)}")
print(f"Entries enhanced by embedding tagging: {entries_enhanced}")
print(f"Total implicit tag additions: {total_implicit_additions}")

# Top 10 entities by implicit additions
if implicit_tag_counts:
    print(f"\nTop 10 entities by implicit tag additions:")
    for tag, count in implicit_tag_counts.most_common(10):
        print(f"  {tag}: +{count} entries")

# -- 6. Build co-occurrence graph -------------------------------------------

edge_entries = defaultdict(list)  # (tag_a, tag_b) -> [entry_ids]
node_entries = defaultdict(list)  # tag -> [entry_ids]

for eid, tags in entry_all_tags.items():
    tags_sorted = sorted(tags)
    for t in tags_sorted:
        node_entries[t].append(eid)
    for i in range(len(tags_sorted)):
        for j in range(i + 1, len(tags_sorted)):
            pair = (tags_sorted[i], tags_sorted[j])
            edge_entries[pair].append(eid)

# Filter edges: weight >= 2
filtered_edges = {
    pair: eids for pair, eids in edge_entries.items()
    if len(eids) >= MIN_EDGE_WEIGHT
}

# Build nodes list
nodes = []
for tag in sorted(node_entries.keys()):
    info = tag_class.get(tag, {"type": "concept", "count": 0})
    nodes.append({
        "id": tag,
        "label": tag.replace("-", " ").title() if len(tag) <= 3 else tag.title(),
        "type": info["type"],
        "entryCount": len(set(node_entries[tag])),
    })

# Build edges list
edges = []
for (src, tgt), eids in sorted(filtered_edges.items(), key=lambda x: -len(x[1])):
    unique_eids = list(set(eids))
    edges.append({
        "source": src,
        "target": tgt,
        "weight": len(unique_eids),
        "entryIds": unique_eids[:50],
    })

graph = {"nodes": nodes, "edges": edges}

# -- 7. Build similarity.json -----------------------------------------------

print(f"\nComputing entity-entity similarity...")
entity_sim = entity_refs @ entity_refs.T
entity_sim = np.nan_to_num(entity_sim, nan=0.0, posinf=0.0, neginf=0.0)

similarity = {}
for i, eid_i in enumerate(entity_ids):
    scores = entity_sim[i]
    # Get top-K excluding self
    top_indices = np.argsort(scores)[::-1]
    similar = []
    for j in top_indices:
        if j == i:
            continue
        if len(similar) >= args.top_k:
            break
        similar.append({
            "id": entity_ids[j],
            "score": round(float(scores[j]), 2),
        })
    similarity[eid_i] = similar

# -- 8. Build cleaned entries for entries.json --------------------------------

cleaned_entries = {}
for eid, tags in entry_all_tags.items():
    raw = raw_by_id.get(eid)
    if raw is None:
        continue

    cleaned_entries[eid] = {
        "id": eid,
        "event": raw["event_name"],
        "date": raw["event_date"],
        "tags": sorted(tags),
        "lines": [
            {"speaker": line["speaker"], "text": strip_html(line["text"])}
            for line in raw["lines"]
        ],
        "note": strip_html(raw.get("note", "")),
    }

# -- 9. Save all three output files ------------------------------------------

with open(data_dir / "graph.json", "w") as f:
    json.dump(graph, f)

with open(data_dir / "entries.json", "w") as f:
    json.dump(cleaned_entries, f)

with open(data_dir / "similarity.json", "w") as f:
    json.dump(similarity, f)

# -- 10. Print comprehensive summary ------------------------------------------

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Model: {args.model}")
print(f"  Threshold: {args.threshold}")
print(f"  Dimensions: {embeddings.shape[1]}")
print(f"  Top-K: {args.top_k}")

print(f"\n  Nodes: {len(nodes)}")
print(f"  Edges: {len(edges)}")
print(f"  Baseline isolated: {len(baseline_isolated)}")

# Count isolated nodes
connected_ids = set()
for e in edges:
    connected_ids.add(e["source"])
    connected_ids.add(e["target"])
isolated = sum(1 for n in nodes if n["id"] not in connected_ids)
print(f"  Isolated nodes now: {isolated}")
print(f"  Nodes rescued: {len(baseline_isolated) - len(baseline_isolated & {n['id'] for n in nodes if n['id'] not in connected_ids})}")

# Nodes by type
type_counts = Counter(n["type"] for n in nodes)
print(f"\n  Nodes by type:")
for t, c in type_counts.most_common():
    print(f"    {t}: {c}")

# Entity reference embedding stats
print(f"\n  Entities with reference embeddings: {len(entity_ids)}")
print(f"  Entities without (< {MIN_ENTRIES_FOR_REF} entries): {len(skipped_entities)}")

# Implicit tag stats
print(f"\n  Entries enhanced by embedding: {entries_enhanced}")
print(f"  Total implicit tags added: {total_implicit_additions}")
if total_implicit_additions > 0 and entries_enhanced > 0:
    print(f"  Avg implicit tags per enhanced entry: {total_implicit_additions / entries_enhanced:.1f}")

# Sample similarity lists
print(f"\n  Sample similarity lists:")
for sample_name in ["kaladin", "hoid", "kelsier"]:
    if sample_name in similarity:
        top3 = similarity[sample_name][:3]
        pairs = ", ".join(f"{s['id']} ({s['score']})" for s in top3)
        print(f"    {sample_name}: {pairs}")
    else:
        print(f"    {sample_name}: (not in entity refs)")

# Saved files
print(f"\n  Saved files:")
for fname in ["graph.json", "entries.json", "similarity.json"]:
    fpath = data_dir / fname
    size = fpath.stat().st_size
    print(f"    {fname}: {size / 1024 / 1024:.1f} MB")

print(f"\n  Entries in entries.json: {len(cleaned_entries)}")
print(f"  Entities in similarity.json: {len(similarity)}")
print()
