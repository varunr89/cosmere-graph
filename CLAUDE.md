# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive force-directed graph visualization of Brandon Sanderson's Cosmere universe, built from 16,282 "Words of Brandon" (WoB) Q&A entries from Arcanum. The frontend uses ES modules (no build step) with D3.js; the data pipeline is a 7-stage Python script sequence with shared utilities in `notebooks/common/`.

## Commands

### Running the Frontend

Serve statically -- no build step required:
```bash
make serve                                            # python3 -m http.server 8000
```

### Tests

```bash
make test                                             # Playwright E2E tests (all specs)
make test-python                                      # Python integration tests
make test-all                                         # both
npx playwright test tests/test_e2e.spec.js            # single spec
```

Playwright uses `playwright.config.js` which auto-starts a server on port 8080 via `webServer`. Shared test helpers are in `tests/helpers.js`. Python test fixtures are in `tests/conftest.py`.

### Data Pipeline

Scripts run sequentially from `notebooks/` using the venv Python:
```bash
python notebooks/01_classify_tags.py
python notebooks/02_build_graph.py
python notebooks/03_text_matching.py
python notebooks/04_embed_entries.py                 # all models
python notebooks/04_embed_entries.py --models voyage,gemini  # subset
python notebooks/05_build_graph.py --model gemini --threshold 0.6
python notebooks/06_compare_models.py
python notebooks/07_export_scores.py --model azure_openai --floor 0.70
```

Pipeline scripts expect `../../words-of-brandon/wob_entries.json` as the raw data source and API keys in `.env`.

### Python Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

### Frontend (ES modules, no build step)

- **`index.html`** -- HTML structure only (~310 lines). Loads CSS via `<link>` tags and JS via `<script type="module">`.
- **`css/`** -- 5 CSS files split by concern: `theme.css` (variables, scrollbars), `animations.css` (loading, particles), `graph.css` (SVG, header, tooltip), `layout.css` (search, filters, panel, responsive), `components.css` (embedding controls, review, guide modal).
- **`js/app.js`** -- Entry point module. Orchestrates init, data loading, and module wiring.
- **`js/state.js`** -- Shared mutable state with setter functions (ES module exports are read-only bindings).
- **`js/constants.js`** -- Gemstone color maps and type labels.
- **`js/graph.js`** -- D3 force simulation setup (`buildGraph`).
- **`js/panel.js`** -- Side panel, focus/unfocus, node details, entry display.
- **`js/filters.js`** -- Type filter buttons and filter application.
- **`js/search.js`** -- Search input, autocomplete, keyboard shortcuts.
- **`js/embeddings.js`** -- Embedding controls, score loading, implicit tag computation.
- **`js/hypothesis.js`** -- Implicit edge rendering and edge layer toggle.
- **`js/review.js`** -- Review panel for confirming/rejecting implicit tags.
- **`js/particles.js`** -- StormParticles animation and gemstone path generator.
- **`js/mobile.js`** -- Mobile backdrop helpers.
- **`js/guide.js`** -- Worldhopper's Guide modal.
- **`js/tagging-engine.js`** -- Pure-function engine for implicit tag computation (global script, not ES module). Zero DOM dependencies, testable standalone.
- **`data/scores.json`** (8 MB) -- Precomputed per-entity similarity scores loaded by the frontend. Structure: `{ entities: { "kaladin": { specificity, prototypes, calibration: {p10..p50}, scores: {entryId: [sim1, sim2]} } } }`.
- **`data/graph.json`** -- Nodes (783 entities, 5 types) and co-occurrence edges (10,696).
- **`data/entries.json`** -- 12,562 cleaned WoB entries with tags and text.

### Entity Types and Theme Colors

| Type         | Color Variable    | Hex       |
|-------------|-------------------|-----------|
| Character   | `--gem-sapphire`  | `#2E5BA8` |
| World       | `--gem-emerald`   | `#2D8B57` |
| Magic System| `--gem-amethyst`  | `#7B4BAA` |
| Shard       | `--gem-heliodor`  | `#C49A2A` |
| Concept     | `--gem-ruby`      | `#B03A3A` |

### Data Pipeline (`notebooks/`)

Seven numbered Python scripts that run in order. Shared utilities live in `notebooks/common/`:
- **`common/paths.py`** -- Centralized path constants (`PROJECT_ROOT`, `DATA_DIR`, `WOB_PATH`, etc.)
- **`common/models.py`** -- Shared constants (`ALL_MODELS`, `EXCLUDE_TYPES`, `MIN_EDGE_WEIGHT`, etc.)
- **`common/html_utils.py`** -- `strip_html()` using `html.unescape`
- **`common/graph_builder.py`** -- `build_cooccurrence_edges()`, `build_nodes()`, `build_edges()`
- **`common/embeddings.py`** -- `load_embeddings()`, `normalize_embeddings()`, `compute_entity_refs()`

Pipeline scripts:
1. **01_classify_tags.py** -- Classifies 886 WoB tags into entity types (character, world, magic, shard, concept, book, meta).
2. **02_build_graph.py** -- Builds co-occurrence graph from classified tags. Outputs `graph.json` and `entries.json`.
3. **03_text_matching.py** -- Regex-based entity matching in entry text to find implicit co-occurrences.
4. **04_embed_entries.py** -- Embeds all 16,282 entries with up to 5 models (Azure OpenAI, Cohere, Mistral, Gemini, Voyage). Caches to `data/embeddings_cache/*.npy`.
5. **05_build_graph.py** -- Builds knowledge graph using embedding cosine-similarity instead of regex.
6. **06_compare_models.py** -- Evaluates models on disambiguation, similarity, and discovery tasks using hand-labeled ground truth.
7. **07_export_scores.py** -- Exports `scores.json` with multi-prototype reference embeddings (via k-means), calibration percentiles, and specificity scores for the frontend.

### Test Structure

- **`playwright.config.js`** -- Centralized config with `webServer` (auto-starts server on port 8080), timeouts, and retries.
- **`tests/helpers.js`** -- Shared helpers: `waitForAppReady()`, `clickApplyAndWait()`, `getTagCount()`.
- **`tests/conftest.py`** -- Shared pytest fixtures for project paths and sample data.
- **`tests/test_e2e.spec.js`** -- Full embedding controls workflow (Playwright).
- **`tests/test_controls.spec.js`** -- Embedding controls bar rendering and Apply button.
- **`tests/test_hypothesis.spec.js`** -- Implicit (dashed amber) vs co-occurrence (solid) edge rendering.
- **`tests/test_tagging_engine.spec.js`** -- Tagging engine pure function tests via Playwright.
- **`tests/test_tagging_engine.html`** -- Standalone HTML unit test runner for the tagging engine.
- **`tests/test_export_scores.py`** -- Integration tests for `07_export_scores.py` using real embedding data (skips if `azure_openai.npy` is absent).
- **`tests/fixtures/sample_scores.json`** -- Sample test data with 4 entities.

### Key Data Flow

```
wob_entries.json -> classify -> build graph -> embed entries -> build similarity graph -> export scores
                                    |                                                        |
                                    v                                                        v
                               graph.json + entries.json                               scores.json
                                    \                                                      /
                                     \----------> index.html (frontend) <-----------------/
```

### External Dependencies

- Raw WoB data lives at `../../words-of-brandon/wob_entries.json` (sibling repo).
- Embedding API keys (Azure OpenAI, Cohere, Gemini, Voyage) are in `.env`.
- Cached embeddings in `data/embeddings_cache/` are large (445 MB total) and gitignored.
