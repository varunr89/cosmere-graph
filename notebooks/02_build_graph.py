"""
Build the co-occurrence graph from classified WoB tags.

Outputs:
- data/graph.json: nodes + edges for the knowledge graph
- data/entries.json: cleaned WoB entries for the frontend
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
wob_path = Path(__file__).parent.parent.parent / "words-of-brandon" / "wob_entries.json"

# ── Load data ───────────────────────────────────────────────────────────────

with open(wob_path) as f:
    entries = json.load(f)

with open(data_dir / "tag_classifications.json") as f:
    tag_class = json.load(f)

# Entity tags only (exclude meta)
entity_tags = {t for t, info in tag_class.items() if info["type"] != "meta"}

print(f"Total entries: {len(entries)}")
print(f"Entity tags: {len(entity_tags)}")

# ── Strip HTML from entry text ──────────────────────────────────────────────

def strip_html(text):
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    return text.strip()

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

# ── Build co-occurrence matrix ──────────────────────────────────────────────

# For each pair of entity tags that appear on the same entry, track the entry IDs
edge_entries = defaultdict(list)  # (tag_a, tag_b) -> [entry_ids]
node_entries = defaultdict(list)  # tag -> [entry_ids]

for e in entries:
    eid = e["id"]
    tags = sorted(set(t for t in e["tags"] if t in entity_tags))
    for t in tags:
        node_entries[t].append(eid)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            pair = (tags[i], tags[j])
            edge_entries[pair].append(eid)

print(f"Nodes: {len(node_entries)}")
print(f"Raw edges: {len(edge_entries)}")

# ── Filter edges ────────────────────────────────────────────────────────────

# Only keep edges with at least 2 co-occurrences to reduce noise
MIN_EDGE_WEIGHT = 2
filtered_edges = {
    pair: eids for pair, eids in edge_entries.items()
    if len(eids) >= MIN_EDGE_WEIGHT
}
print(f"Edges (weight >= {MIN_EDGE_WEIGHT}): {len(filtered_edges)}")

# ── Build graph JSON ────────────────────────────────────────────────────────

nodes = []
for tag in sorted(node_entries.keys()):
    info = tag_class.get(tag, {"type": "concept", "count": 0})
    nodes.append({
        "id": tag,
        "label": tag.replace("-", " ").title() if len(tag) <= 3 else tag.title(),
        "type": info["type"],
        "entryCount": len(node_entries[tag]),
    })

edges = []
for (src, tgt), eids in sorted(filtered_edges.items(), key=lambda x: -len(x[1])):
    edges.append({
        "source": src,
        "target": tgt,
        "weight": len(eids),
        "entryIds": eids[:50],  # Cap at 50 entry IDs per edge for file size
    })

graph = {"nodes": nodes, "edges": edges}

# ── Summary stats ───────────────────────────────────────────────────────────

type_node_counts = Counter(n["type"] for n in nodes)
print(f"\nNodes by type:")
for t, c in type_node_counts.most_common():
    print(f"  {t}: {c}")

print(f"\nTop 20 edges by weight:")
for e in edges[:20]:
    print(f"  {e['source']} ↔ {e['target']}: {e['weight']}")

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
