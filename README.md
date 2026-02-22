# Cosmere Knowledge Graph

Interactive force-directed graph visualization of Brandon Sanderson's Cosmere, built from **16,000+ Words of Brandon** Q&A entries from [Arcanum](https://wob.coppermind.net/).

**[View the live visualization](https://varunr89.github.io/cosmere-graph/)**

## What is this?

Every node is a Cosmere entity (character, world, magic system, Shard, or concept). Connections form when two entities are discussed in the same Q&A entry. Thicker connections mean more shared entries.

- **783 entity nodes** across 5 types
- **10,696 weighted connections**
- **12,562 searchable WoB entries**

## Features

- Search for any entity (press `/` to focus)
- Click a node to focus and see its connections
- Click a connection to read the actual Q&A entries
- "Browse all entries" to read every WoB entry for any entity
- Filter by entity type (Characters, Worlds, Magic Systems, Shards, Concepts)
- Embedding-based implicit tag discovery with tunable controls
- Review panel to confirm/reject implicit tags
- Mobile responsive with full touch support
- Worldhopper's Guide onboarding modal

## Entity types

| Type | Gemstone | Examples |
|------|----------|----------|
| Characters | Sapphire | Hoid, Kaladin, Vin, Kelsier |
| Worlds | Emerald | Roshar, Scadrial, Sel, Nalthis |
| Magic Systems | Amethyst | Allomancy, Surgebinding, Awakening |
| Shards | Heliodor | Honor, Odium, Preservation, Ruin |
| Concepts | Ruby | Investiture, Knights Radiant, Desolations |

## Data pipeline

1. All 16,282 WoB entries fetched from the Arcanum API
2. 886 tags classified into entity types via LLM (`notebooks/01_classify_tags.py`)
3. Co-occurrence graph built from explicit tag pairs (`notebooks/02_build_graph.py`)
4. Text-based entity matching scans entry text for implicit mentions (`notebooks/03_text_matching.py`)
5. Embedding vectors computed via OpenAI, Cohere, and Gemini models (`notebooks/04_embed_entries.py`)
6. Embedding-based similarity graph built with calibrated thresholds (`notebooks/05_build_graph.py`)
7. Model comparison across 3 embedding providers (`notebooks/06_compare_models.py`)
8. Final scores exported for the frontend (`notebooks/07_export_scores.py`)

## Embedding model comparison

Three embedding models were evaluated (see `data/model_comparison.txt`):
- **Azure OpenAI** (text-embedding-3-large, 3072d)
- **Cohere** (embed-english-v3.0, 1024d)
- **Google Gemini** (text-embedding-004, 768d)

Disambiguation ground truth in `data/disambiguation_ground_truth.json` was used to evaluate precision.

## Running locally

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

To regenerate the data pipeline:

```bash
pip install -r requirements.txt
python3 notebooks/01_classify_tags.py
python3 notebooks/02_build_graph.py
# ... through 07_export_scores.py
```

To run tests:

```bash
npm install
npx playwright test tests/ --reporter=list
```

## Credits

- Data: [Arcanum / Coppermind](https://wob.coppermind.net/)
- Visualization: D3.js force-directed graph
- Theme: Stormlight Archive Knight Radiant gemstone system
