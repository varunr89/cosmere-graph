"""
Build the co-occurrence graph from classified WoB tags.

Outputs:
- data/graph.json: nodes + edges for the knowledge graph
- data/entries.json: cleaned WoB entries for the frontend
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common.paths import DATA_DIR as data_dir, WOB_PATH as wob_path
from common.html_utils import strip_html
from common.models import MIN_EDGE_WEIGHT
from common.graph_builder import build_cooccurrence_edges, build_nodes, build_edges

# ── Load data ───────────────────────────────────────────────────────────────

with open(wob_path) as f:
    entries = json.load(f)

with open(data_dir / "tag_classifications.json") as f:
    tag_class = json.load(f)

# Entity tags only (exclude meta)
entity_tags = {t for t, info in tag_class.items() if info["type"] != "meta"}

print(f"Total entries: {len(entries)}")
print(f"Entity tags: {len(entity_tags)}")

# ── Build cleaned entries ───────────────────────────────────────────────────

cleaned_entries = {}
for e in entries:
    eid = e["id"]
    entity_tags_in_entry = [t for t in e["tags"] if t in entity_tags]
    if not entity_tags_in_entry:
        continue  # Skip entries with no entity tags

    cleaned_entries[eid] = {
        "id": eid,
        "event": e["event_name"],
        "date": e["event_date"],
        "tags": entity_tags_in_entry,
        "lines": [
            {"speaker": line["speaker"], "text": strip_html(line["text"])}
            for line in e["lines"]
        ],
        "note": strip_html(e.get("note", "")),
    }

print(f"Entries with entity tags: {len(cleaned_entries)}")

# ── Build co-occurrence graph ───────────────────────────────────────────────

node_entries, filtered_edges = build_cooccurrence_edges(entries, entity_tags)

print(f"Nodes: {len(node_entries)}")
print(f"Edges (weight >= {MIN_EDGE_WEIGHT}): {len(filtered_edges)}")

# ── Build graph JSON ────────────────────────────────────────────────────────

nodes = build_nodes(node_entries, tag_class)
edges = build_edges(filtered_edges)

graph = {"nodes": nodes, "edges": edges}

# ── Summary stats ───────────────────────────────────────────────────────────

type_node_counts = Counter(n["type"] for n in nodes)
print(f"\nNodes by type:")
for t, c in type_node_counts.most_common():
    print(f"  {t}: {c}")

print(f"\nTop 20 edges by weight:")
for e in edges[:20]:
    print(f"  {e['source']} \u2194 {e['target']}: {e['weight']}")

# ── Save ────────────────────────────────────────────────────────────────────

with open(data_dir / "graph.json", "w") as f:
    json.dump(graph, f)
print(f"\nSaved graph.json: {len(nodes)} nodes, {len(edges)} edges")

# Save entries as a dict keyed by ID for fast lookup
with open(data_dir / "entries.json", "w") as f:
    json.dump(cleaned_entries, f)
print(f"Saved entries.json: {len(cleaned_entries)} entries")

# File sizes
for fname in ["graph.json", "entries.json"]:
    size = (data_dir / fname).stat().st_size
    print(f"  {fname}: {size / 1024 / 1024:.1f} MB")
