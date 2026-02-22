"""
Text-based entity matching: find entity mentions in WoB entry text
to create implicit co-occurrence edges beyond explicit Arcanum tags.

Run AFTER 02_build_graph.py. Rebuilds graph.json and entries.json
with richer connections.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.paths import DATA_DIR as data_dir, WOB_PATH as wob_path
from common.html_utils import strip_html
from common.models import MIN_EDGE_WEIGHT
from common.graph_builder import build_cooccurrence_edges, build_nodes, build_edges

# ── Load data ────────────────────────────────────────────────────────────────

with open(data_dir / "tag_classifications.json") as f:
    tag_class = json.load(f)

with open(wob_path) as f:
    raw_entries = json.load(f)

entity_tags = {t for t, info in tag_class.items() if info["type"] != "meta"}

print(f"Total raw entries: {len(raw_entries)}")
print(f"Entity tags: {len(entity_tags)}")

# ── Merge singular/plural duplicates ─────────────────────────────────────────

# Detect: if both "X" and "Xs" exist as entity tags, merge X -> Xs
merge_map = {}
for tag in sorted(entity_tags):
    plural = tag + "s"
    if plural in entity_tags and tag != plural:
        merge_map[tag] = plural

print(f"\nSingular/plural merges: {len(merge_map)}")
for s, p in merge_map.items():
    print(f"  {s} -> {p}")

def apply_merge(tag):
    return merge_map.get(tag, tag)

# Remove merged-away tags from the entity set
merged_entity_tags = {apply_merge(t) for t in entity_tags} - set(merge_map.keys())

# ── Build text-matching patterns ─────────────────────────────────────────────

# Skip names that are too short or are common English words that cause
# false positives even in Cosmere context
SKIP_TEXT_MATCH = {
    # Very common English words (metals used in Allomancy)
    "iron", "steel", "tin", "brass", "bronze", "copper", "pewter", "zinc",
    "gold", "aluminum", "chromium", "cadmium", "electrum", "platinum",
    "nicrosil", "duralumin", "bendalloy",
    # Common English words that are also entity names
    "felt", "design", "cord", "dusk", "wode",
    # Generic Cosmere terms that would match almost everything
    "cosmere", "magic", "shards",
}

patterns = {}
for tag in merged_entity_tags:
    if tag in SKIP_TEXT_MATCH:
        continue
    if len(tag) < 4:
        continue
    escaped = re.escape(tag)
    patterns[tag] = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)

print(f"Text-match patterns: {len(patterns)}")

# ── Process all entries: explicit tags + text-matched tags ───────────────────

cleaned_entries = {}
entry_all_tags = {}  # eid -> set of all tags (explicit + text-matched)

text_match_additions = 0
entries_enhanced = 0

for e in raw_entries:
    eid = e["id"]

    # Explicit entity tags (with merge applied)
    explicit = set(apply_merge(t) for t in e["tags"] if t in entity_tags)

    # Build full text from all lines
    lines_text = " ".join(strip_html(line["text"]) for line in e["lines"])
    note_text = strip_html(e.get("note", ""))
    full_text = lines_text + " " + note_text

    # Text-match: find entity names mentioned in text
    text_matched = set()
    for tag, pattern in patterns.items():
        if tag in explicit:
            continue
        if pattern.search(full_text):
            text_matched.add(tag)

    all_tags = explicit | text_matched
    # Only include entries with at least one entity tag
    if not all_tags:
        continue

    if text_matched:
        text_match_additions += len(text_matched)
        entries_enhanced += 1

    entry_all_tags[eid] = all_tags

    cleaned_entries[eid] = {
        "id": eid,
        "event": e["event_name"],
        "date": e["event_date"],
        "tags": sorted(all_tags),
        "lines": [
            {"speaker": line["speaker"], "text": strip_html(line["text"])}
            for line in e["lines"]
        ],
        "note": note_text if note_text else "",
    }

print(f"\nEntries with entity tags: {len(cleaned_entries)}")
print(f"Entries enhanced by text matching: {entries_enhanced}")
print(f"Total implicit tag additions: {text_match_additions}")

# ── Build co-occurrence graph ────────────────────────────────────────────────

cooc_entries = [{"id": eid, "tags": list(tags)} for eid, tags in entry_all_tags.items()]
node_entries, filtered_edges = build_cooccurrence_edges(cooc_entries, merged_entity_tags)

# Build nodes
nodes = build_nodes(node_entries, tag_class)

# Build edges
edges = build_edges(filtered_edges)

graph = {"nodes": nodes, "edges": edges}

# ── Compare with previous graph ──────────────────────────────────────────────

with open(data_dir / "graph.json") as f:
    old_graph = json.load(f)

old_nodes = len(old_graph["nodes"])
old_edges = len(old_graph["edges"])

# Count isolated nodes in new graph
connected_ids = set()
for e in edges:
    connected_ids.add(e["source"])
    connected_ids.add(e["target"])
new_isolated = sum(1 for n in nodes if n["id"] not in connected_ids)

print(f"\n── Comparison ──")
print(f"  Old: {old_nodes} nodes, {old_edges} edges")
print(f"  New: {len(nodes)} nodes, {len(edges)} edges")
print(f"  Old isolated: 270")
print(f"  New isolated: {new_isolated}")
print(f"  New edges from text matching: {len(edges) - old_edges}")

# ── Save ─────────────────────────────────────────────────────────────────────

with open(data_dir / "graph.json", "w") as f:
    json.dump(graph, f)
print(f"\nSaved graph.json: {len(nodes)} nodes, {len(edges)} edges")

with open(data_dir / "entries.json", "w") as f:
    json.dump(cleaned_entries, f)
print(f"Saved entries.json: {len(cleaned_entries)} entries")

for fname in ["graph.json", "entries.json"]:
    size = (data_dir / fname).stat().st_size
    print(f"  {fname}: {size / 1024 / 1024:.1f} MB")
