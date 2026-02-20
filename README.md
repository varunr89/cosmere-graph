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
2. 886 tags classified into entity types via LLM
3. Co-occurrence graph built from explicit tag pairs
4. Text-based entity matching scans entry text for implicit mentions, recovering ~8,400 additional connections

## Credits

- Data: [Arcanum / Coppermind](https://wob.coppermind.net/)
- Visualization: D3.js force-directed graph
- Theme: Stormlight Archive Knight Radiant gemstone system
