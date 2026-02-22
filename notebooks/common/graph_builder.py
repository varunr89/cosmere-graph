"""Co-occurrence graph construction utilities."""

from collections import defaultdict

from .models import MIN_EDGE_WEIGHT


def build_cooccurrence_edges(entries, entity_tags, min_weight=MIN_EDGE_WEIGHT):
    """
    Build co-occurrence edges from entry tag lists.

    Args:
        entries: list of dicts or dict of dicts, each with a "tags" key
        entity_tags: set of valid entity tag names to include
        min_weight: minimum number of shared entries to create an edge

    Returns:
        (node_entries, filtered_edges) where:
        - node_entries: dict mapping tag -> list of entry IDs
        - filtered_edges: dict mapping (tag_a, tag_b) -> list of entry IDs
    """
    edge_entries = defaultdict(list)
    node_entries = defaultdict(list)

    items = entries if isinstance(entries, list) else entries.values()
    for entry in items:
        eid = entry["id"] if isinstance(entry, dict) and "id" in entry else entry.get("id")
        tags = sorted(set(t for t in entry.get("tags", []) if t in entity_tags))
        for t in tags:
            node_entries[t].append(eid)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair = (tags[i], tags[j])
                edge_entries[pair].append(eid)

    filtered_edges = {
        pair: eids for pair, eids in edge_entries.items()
        if len(eids) >= min_weight
    }

    return node_entries, filtered_edges


def build_nodes(node_entries, tag_class):
    """
    Build node list from node_entries and tag classifications.

    Args:
        node_entries: dict mapping tag -> list of entry IDs
        tag_class: dict mapping tag -> {"type": str, "count": int}

    Returns:
        list of node dicts with id, label, type, entryCount
    """
    nodes = []
    for tag in sorted(node_entries.keys()):
        info = tag_class.get(tag, {"type": "concept", "count": 0})
        nodes.append({
            "id": tag,
            "label": tag.replace("-", " ").title() if len(tag) <= 3 else tag.title(),
            "type": info["type"],
            "entryCount": len(set(node_entries[tag])),
        })
    return nodes


def build_edges(filtered_edges, cap=50):
    """
    Build edge list from filtered co-occurrence edges.

    Args:
        filtered_edges: dict mapping (tag_a, tag_b) -> list of entry IDs
        cap: max entry IDs to store per edge

    Returns:
        list of edge dicts with source, target, weight, entryIds
    """
    edges = []
    for (a, b), eids in sorted(filtered_edges.items(), key=lambda x: -len(x[1])):
        edges.append({
            "source": a,
            "target": b,
            "weight": len(eids),
            "entryIds": eids[:cap],
        })
    return edges
