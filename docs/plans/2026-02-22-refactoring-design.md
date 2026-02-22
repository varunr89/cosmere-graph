# Refactoring Design: Modularize + Lite Tooling

**Date:** 2026-02-22
**Approach:** Modularize in place with lite tooling (Approach C)

## Goal

Refactor the cosmere-graph codebase for maintainability and developer experience without introducing a build step or heavy tooling. Split monolithic files into modules, extract duplicated code into shared utilities, and add proper test configuration.

## Frontend Modularization

### Current State
- One 3,400-line `index.html` containing all CSS (~1,450 lines), HTML (~290 lines), and JS (~1,660 lines)
- All JavaScript in global scope with mutable shared state
- 18 functional sections separated by comments but no encapsulation
- External: D3.js v7 (CDN), `js/tagging-engine.js` (local)

### Proposed Structure

```
index.html              (~290 lines - HTML only, <link> + <script type="module"> tags)
css/
  theme.css             (~200 lines - CSS variables, fonts, base colors)
  layout.css            (~400 lines - grid, panels, containers)
  components.css        (~500 lines - buttons, inputs, modals, filters)
  graph.css             (~200 lines - SVG styling, nodes, edges)
  animations.css        (~150 lines - loading screen, particles, transitions)
js/
  tagging-engine.js     (exists, unchanged)
  constants.js          (~50 lines  - gemstone colors, type labels, config)
  state.js              (~30 lines  - shared state: focusedNode, activeFilters, etc.)
  particles.js          (~70 lines  - StormParticles class)
  graph.js              (~200 lines - buildGraph, D3 force simulation)
  panel.js              (~400 lines - side panel, node details, entry display)
  search.js             (~100 lines - search input, autocomplete)
  filters.js            (~60 lines  - filter buttons, toggle logic)
  embeddings.js         (~150 lines - embedding controls, sliders, apply)
  hypothesis.js         (~120 lines - implicit edge rendering, edge layer toggle)
  review.js             (~250 lines - review panel, confirm/reject)
  app.js                (~50 lines  - init(), data loading, wiring modules together)
```

### Key Decisions
- ES modules (`import`/`export`) for explicit dependency tracking (requires HTTP server, already standard workflow)
- Shared state in `state.js` as named exports rather than globals
- `app.js` is the entry point (`<script type="module" src="js/app.js">`)
- CSS split by concern (theme/layout/components/graph/animations)

## Python Pipeline Refactoring

### Current State
- 7 numbered scripts in `notebooks/` (2,510 total lines)
- `strip_html()` duplicated 5 times (with inconsistent implementations)
- Graph-building logic duplicated 4 times
- Embedding normalization duplicated 3 times
- Path construction duplicated 7 times
- `ALL_MODELS` list in 4 places

### Proposed Structure

```
notebooks/
  common/
    __init__.py
    paths.py            (~30 lines  - PROJECT_ROOT, DATA_DIR, WOB_PATH, CACHE_DIR)
    html_utils.py       (~15 lines  - strip_html using html.unescape)
    graph_builder.py    (~80 lines  - build_cooccurrence_graph, build_nodes, build_edges)
    embeddings.py       (~60 lines  - load_embeddings, normalize_embeddings, compute_entity_refs)
    models.py           (~20 lines  - ALL_MODELS, EXCLUDE_TYPES, MIN_EDGE_WEIGHT, etc.)
  01_classify_tags.py   (import paths from common)
  02_build_graph.py     (uses graph_builder, html_utils)
  03_text_matching.py   (uses graph_builder, html_utils)
  04_embed_entries.py   (uses paths, models)
  05_build_graph.py     (uses graph_builder, embeddings, html_utils)
  06_compare_models.py  (uses embeddings, graph_builder)
  07_export_scores.py   (uses embeddings, paths, models)
```

### Shared Module Contents
- **paths.py**: PROJECT_ROOT, DATA_DIR, WOB_PATH, CACHE_DIR, EMBEDDINGS_CACHE_DIR
- **html_utils.py**: Single `strip_html()` using `html.unescape()` (most correct version)
- **graph_builder.py**: `build_cooccurrence_edges()`, `build_nodes()`, `filter_edges()`
- **embeddings.py**: `load_embeddings()`, `normalize_embeddings()`, `compute_entity_refs()`
- **models.py**: `ALL_MODELS`, `EXCLUDE_TYPES`, `MIN_EDGE_WEIGHT`, `DEFAULT_THRESHOLD`

### What Does NOT Change
- Each script's CLI interface and argparse arguments
- Pipeline execution order and data flow
- Output file formats and content

## Test Infrastructure

### Current State
- 50+ lines of HTTP server setup copy-pasted across 4 Playwright specs
- No `playwright.config.js`
- No shared test helpers
- No `conftest.py` for Python tests
- Fragile 90-second timeouts

### Proposed Changes

**playwright.config.js**: Centralized config with `webServer` directive to manage the HTTP server lifecycle, base URL, timeouts, and retry policy. Eliminates duplicated server setup from all 4 test files.

**tests/helpers.js**: Shared `waitForAppReady()`, `clickApplyAndWait()`, `getTagCount()` helpers extracted from duplicated code across specs.

**tests/conftest.py**: Shared pytest fixtures for project paths and sample data loading.

## Tooling

**Makefile** with targets:
- `serve` - Start local HTTP server
- `test` - Run Playwright E2E tests
- `test-python` - Run pytest integration tests
- `test-all` - Run all tests

## Scope Summary

| Area | New Files | Modified Files |
|------|-----------|----------------|
| Frontend | 17 (12 JS + 5 CSS) | 1 (index.html) |
| Python | 6 (common/ package) | 7 (all notebook scripts) |
| Tests | 3 (config, helpers, conftest) | 4 (all Playwright specs) |
| Tooling | 1 (Makefile) | 0 |

## What Does NOT Change
- D3.js visualization logic (just moves between files)
- Python script behavior and CLI interfaces
- Test assertions and coverage
- Data files and pipeline outputs
- No build step introduced

## Risk Mitigation
- Run existing Playwright tests after each major step
- Frontend split is purely mechanical (move code between files)
- Python scripts are updated one at a time with test verification
