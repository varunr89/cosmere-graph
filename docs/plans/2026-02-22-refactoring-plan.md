# Codebase Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the cosmere-graph codebase for maintainability and developer experience -- split the monolithic index.html into ES modules, extract duplicated Python utilities, and add proper test configuration.

**Architecture:** The frontend stays a no-build-step static site but moves from one 3,400-line HTML file to separate CSS and JS modules loaded via `<link>` and `<script type="module">`. The Python pipeline extracts shared code into `notebooks/common/`. Test infra gains a `playwright.config.js` and shared helpers.

**Tech Stack:** Vanilla JS (ES modules), D3.js v7, Python 3, Playwright, pytest, Make

---

## Phase 1: Test Infrastructure (do this first so we can verify everything after)

### Task 1: Create playwright.config.js

**Files:**
- Create: `playwright.config.js`

**Step 1: Create the config file**

```js
// playwright.config.js
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 120_000,
  retries: 1,
  reporter: 'list',
  webServer: {
    command: 'python3 -m http.server 8080 --bind 127.0.0.1',
    port: 8080,
    reuseExistingServer: true,
  },
  use: {
    baseURL: 'http://127.0.0.1:8080',
  },
});
```

**Step 2: Verify tests still pass with the new config**

Run: `npx playwright test tests/test_controls.spec.js --reporter=list`
Expected: All tests pass (the old inline server still works; we haven't removed it yet)

**Step 3: Commit**

```bash
git add playwright.config.js
git commit -m "chore: add playwright.config.js with webServer"
```

---

### Task 2: Create shared test helpers

**Files:**
- Create: `tests/helpers.js`

**Step 1: Extract shared helpers from test files**

The following functions are duplicated across `test_e2e.spec.js`, `test_controls.spec.js`, and `test_hypothesis.spec.js`. Create a single shared module.

```js
// tests/helpers.js

/**
 * Wait for the app to finish loading (loading screen fades, graph-container visible).
 * Used by: test_e2e, test_controls, test_hypothesis
 */
async function waitForAppReady(page) {
  await page.waitForFunction(() => {
    var gc = document.getElementById('graph-container');
    return gc && gc.style.opacity === '1';
  }, { timeout: 30000 });
}

/**
 * Click the Apply button and wait for implicit tag computation to finish.
 * Used by: test_e2e, test_hypothesis
 */
async function clickApplyAndWait(page) {
  await page.evaluate(function() {
    window._lastImplicitResult = null;
  });
  await page.locator('#apply-embeddings-btn').click();
  await page.waitForFunction(function() {
    return window._lastImplicitResult && window._lastImplicitResult.edges;
  }, { timeout: 90000 });
}

/**
 * Get the current implicit tag count from the stats display.
 * Used by: test_controls
 */
async function getTagCount(page) {
  return page.evaluate(function() {
    var text = document.getElementById('embedding-stats').textContent;
    var match = text.match(/(\d+)\s+tags/);
    return match ? parseInt(match[1], 10) : 0;
  });
}

module.exports = { waitForAppReady, clickApplyAndWait, getTagCount };
```

**Step 2: Commit**

```bash
git add tests/helpers.js
git commit -m "chore: extract shared test helpers"
```

---

### Task 3: Remove duplicated server setup from all 4 Playwright specs

Now that `playwright.config.js` provides a `webServer`, remove the inline `createServer()`, `test.beforeAll`, and `test.afterAll` blocks from each test file. Replace `serverInfo.url` with empty string (since `baseURL` is set).

**Files:**
- Modify: `tests/test_controls.spec.js` -- remove lines 1-59 (server + helpers), add require of helpers.js
- Modify: `tests/test_e2e.spec.js` -- remove lines 1-70 (server + helpers), add require of helpers.js
- Modify: `tests/test_hypothesis.spec.js` -- remove server setup, add require of helpers.js
- Modify: `tests/test_tagging_engine.spec.js` -- remove server setup

**Step 1: Update test_controls.spec.js**

Replace the file header (everything before the first `test(` call) with:
```js
const { test, expect } = require('@playwright/test');
const { waitForAppReady, getTagCount } = require('./helpers');
```

Replace all `serverInfo.url + '/index.html'` with `'/index.html'`.

Remove the `waitForAppReady` function definition (now imported).

**Step 2: Update test_e2e.spec.js**

Replace the file header with:
```js
const { test, expect } = require('@playwright/test');
const { waitForAppReady, clickApplyAndWait } = require('./helpers');
```

Replace all `serverInfo.url + '/index.html'` with `'/index.html'`.

Remove the `waitForAppReady` and `clickApplyAndWait` function definitions (now imported).

**Step 3: Update test_hypothesis.spec.js**

Same pattern. Import `{ waitForAppReady, clickApplyAndWait }` from helpers. Remove duplicated functions and server setup. Replace URL references.

**Step 4: Update test_tagging_engine.spec.js**

Same pattern. Import needed helpers. Remove server setup. Replace URL references. Note: this file navigates to `'/tests/test_tagging_engine.html'` rather than `'/index.html'`.

**Step 5: Run all tests**

Run: `npx playwright test tests/ --reporter=list`
Expected: All tests pass. The server is now managed by playwright.config.js.

**Step 6: Commit**

```bash
git add tests/test_controls.spec.js tests/test_e2e.spec.js tests/test_hypothesis.spec.js tests/test_tagging_engine.spec.js
git commit -m "refactor: remove duplicated server setup from test files"
```

---

### Task 4: Create conftest.py and Makefile

**Files:**
- Create: `tests/conftest.py`
- Create: `Makefile`

**Step 1: Create conftest.py**

```python
"""Shared pytest fixtures for cosmere-graph tests."""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "embeddings_cache"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def sample_scores():
    with open(FIXTURES_DIR / "sample_scores.json") as f:
        return json.load(f)
```

**Step 2: Create Makefile**

```makefile
.PHONY: serve test test-python test-all

serve:
	python3 -m http.server 8000

test:
	npx playwright test tests/ --reporter=list

test-python:
	python -m pytest tests/test_export_scores.py -v

test-all: test test-python
```

**Step 3: Verify**

Run: `make test`
Expected: Playwright tests pass.

**Step 4: Commit**

```bash
git add tests/conftest.py Makefile
git commit -m "chore: add conftest.py and Makefile"
```

---

## Phase 2: Python Pipeline Refactoring

### Task 5: Create notebooks/common/ package with paths and constants

**Files:**
- Create: `notebooks/common/__init__.py`
- Create: `notebooks/common/paths.py`
- Create: `notebooks/common/models.py`

**Step 1: Create the package**

```python
# notebooks/common/__init__.py
```

```python
# notebooks/common/paths.py
"""Centralized path constants for the data pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "embeddings_cache"
WOB_PATH = PROJECT_ROOT.parent / "words-of-brandon" / "wob_entries.json"
TAG_CLASS_PATH = DATA_DIR / "tag_classifications.json"
```

```python
# notebooks/common/models.py
"""Shared constants for the data pipeline."""

ALL_MODELS = ["azure_openai", "azure_cohere", "azure_mistral", "gemini", "voyage"]

EXCLUDE_TYPES = {"meta", "book"}

MIN_EDGE_WEIGHT = 2

DEFAULT_THRESHOLD = 0.5

DEFAULT_FLOOR = 0.60
```

**Step 2: Commit**

```bash
git add notebooks/common/
git commit -m "refactor: create notebooks/common/ with paths and constants"
```

---

### Task 6: Create html_utils.py

**Files:**
- Create: `notebooks/common/html_utils.py`

**Step 1: Create the module**

Use `html.unescape()` (the correct approach from `04_embed_entries.py`) rather than the manual entity replacements in `02` and `03`.

```python
# notebooks/common/html_utils.py
"""HTML text cleaning utilities."""

import re
from html import unescape


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from WoB entry text."""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()
```

**Step 2: Commit**

```bash
git add notebooks/common/html_utils.py
git commit -m "refactor: add shared strip_html using html.unescape"
```

---

### Task 7: Create graph_builder.py

**Files:**
- Create: `notebooks/common/graph_builder.py`

**Step 1: Create the module**

Extract the co-occurrence graph building logic duplicated across `02_build_graph.py:65-130`, `03_text_matching.py:139-180`, `05_build_graph.py:287-365`, and `06_compare_models.py:512-560`.

```python
# notebooks/common/graph_builder.py
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
```

**Step 2: Commit**

```bash
git add notebooks/common/graph_builder.py
git commit -m "refactor: add shared graph_builder module"
```

---

### Task 8: Create embeddings.py

**Files:**
- Create: `notebooks/common/embeddings.py`

**Step 1: Create the module**

Extract from `05_build_graph.py:86-174`, `06_compare_models.py:171-211`, and `07_export_scores.py:130-196`.

```python
# notebooks/common/embeddings.py
"""Embedding loading, normalization, and entity reference computation."""

import json

import numpy as np

from .paths import CACHE_DIR


def load_embeddings(model_name):
    """
    Load cached embeddings and entry IDs for a model.

    Args:
        model_name: e.g. "azure_openai", "gemini"

    Returns:
        (embeddings, entry_ids) where embeddings is an ndarray and
        entry_ids is a list of string IDs aligned with rows.

    Raises:
        FileNotFoundError: if the .npy or entry_ids.json file is missing
    """
    npy_path = CACHE_DIR / f"{model_name}.npy"
    ids_path = CACHE_DIR / "entry_ids.json"

    if not npy_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {npy_path}. "
            f"Run 04_embed_entries.py --models {model_name} first."
        )
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Entry IDs not found at {ids_path}. "
            f"Run 04_embed_entries.py first."
        )

    embeddings = np.load(npy_path)
    with open(ids_path) as f:
        entry_ids = json.load(f)

    if embeddings.shape[0] != len(entry_ids):
        raise ValueError(
            f"Shape mismatch: {npy_path.name} has {embeddings.shape[0]} rows "
            f"but entry_ids.json has {len(entry_ids)} entries"
        )

    return embeddings, entry_ids


def normalize_embeddings(embeddings):
    """
    L2-normalize embeddings, handling zero-norm rows safely.

    Args:
        embeddings: 2D ndarray (n_entries, dims)

    Returns:
        Normalized ndarray of same shape. Zero-norm rows become zero vectors.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_mask = (norms.squeeze() == 0)
    zero_count = int(zero_mask.sum())
    if zero_count:
        print(f"  WARNING: {zero_count} entries have zero-norm embeddings")
    norms = np.where(norms == 0, 1.0, norms)
    result = embeddings / norms
    result[zero_mask] = 0.0
    return result


def compute_entity_refs(embeddings, entity_tags, entry_explicit_tags, eid_to_idx,
                        min_entries=5):
    """
    Compute mean reference embeddings for each entity tag.

    For each entity, averages the embeddings of its explicitly-tagged entries
    and L2-normalizes the result.

    Args:
        embeddings: 2D ndarray (n_entries, dims)
        entity_tags: set of entity tag names
        entry_explicit_tags: dict mapping entry_id -> set/list of tag names
        eid_to_idx: dict mapping entry_id -> row index in embeddings
        min_entries: minimum tagged entries required to produce a reference

    Returns:
        dict mapping tag_name -> normalized reference embedding (1D ndarray)
    """
    refs = {}
    for tag in sorted(entity_tags):
        tag_eids = [eid for eid, tags in entry_explicit_tags.items()
                    if tag in tags and eid in eid_to_idx]
        if len(tag_eids) < min_entries:
            continue
        indices = [eid_to_idx[eid] for eid in tag_eids]
        tag_embeddings = embeddings[indices]
        ref = tag_embeddings.mean(axis=0)
        norm = np.linalg.norm(ref)
        if norm > 0:
            ref = ref / norm
        refs[tag] = ref
    return refs
```

**Step 2: Commit**

```bash
git add notebooks/common/embeddings.py
git commit -m "refactor: add shared embeddings module"
```

---

### Task 9: Update 02_build_graph.py to use common modules

**Files:**
- Modify: `notebooks/02_build_graph.py`

**Step 1: Replace duplicated code**

Replace the path setup (lines 14-15), `strip_html` function (lines 33-38), co-occurrence building (lines 65-89), node building (lines 92-104), and edge building (lines 106-125) with imports from `common`.

The updated script should:
1. Import from `common.paths`, `common.html_utils`, `common.graph_builder`, `common.models`
2. Keep the entry loading and cleaning logic (the `for e in entries` loop that builds cleaned entries)
3. Use `build_cooccurrence_edges()`, `build_nodes()`, `build_edges()` from graph_builder
4. Keep the file-writing logic at the bottom

**Step 2: Run quick sanity check**

Run: `cd /Users/varunr/projects/cosmere-graph && python notebooks/02_build_graph.py 2>&1 | head -20`
Expected: Script runs and prints node/edge counts matching the original output. Output files `data/graph.json` and `data/entries.json` should be regenerated.

**Step 3: Commit**

```bash
git add notebooks/02_build_graph.py
git commit -m "refactor: 02_build_graph uses common modules"
```

---

### Task 10: Update 03_text_matching.py to use common modules

**Files:**
- Modify: `notebooks/03_text_matching.py`

**Step 1: Replace duplicated code**

Replace path setup (lines 14-15), `strip_html` (lines 77-81), co-occurrence building (lines 139-156), node/edge building. Import from `common.paths`, `common.html_utils`, `common.graph_builder`, `common.models`.

Keep the text-matching regex logic (the core of this script) -- that's unique to this file.

**Step 2: Commit**

```bash
git add notebooks/03_text_matching.py
git commit -m "refactor: 03_text_matching uses common modules"
```

---

### Task 11: Update 04_embed_entries.py to use common modules

**Files:**
- Modify: `notebooks/04_embed_entries.py`

**Step 1: Replace duplicated code**

Replace path setup (lines 29-37), `ALL_MODELS` (line 41), `strip_html` (lines 60-65). Import from `common.paths`, `common.models`, `common.html_utils`.

Keep all the embedding model functions (they're unique to this script) and the argparse/CLI logic.

**Step 2: Commit**

```bash
git add notebooks/04_embed_entries.py
git commit -m "refactor: 04_embed_entries uses common modules"
```

---

### Task 12: Update 05_build_graph.py to use common modules

**Files:**
- Modify: `notebooks/05_build_graph.py`

**Step 1: Replace duplicated code**

This script has the most duplication. Replace:
- Path setup (lines 37-40)
- `ALL_MODELS` (line 44)
- `strip_html` (lines 77-82)
- `EXCLUDE_TYPES` and entity_tags filtering (lines 125-126)
- Embedding loading and normalization (lines 86-115) with `load_embeddings()` + `normalize_embeddings()`
- Entity reference computation (lines 146-174) with `compute_entity_refs()`
- Co-occurrence building (lines 287-303) with `build_cooccurrence_edges()`
- `MIN_EDGE_WEIGHT` (line 191)

Keep: the embedding-based tagging logic (the core innovation of this script), similarity.json output, and the merged graph construction.

**Step 2: Commit**

```bash
git add notebooks/05_build_graph.py
git commit -m "refactor: 05_build_graph uses common modules"
```

---

### Task 13: Update 06_compare_models.py to use common modules

**Files:**
- Modify: `notebooks/06_compare_models.py`

**Step 1: Replace duplicated code**

Replace:
- Path setup (lines 34-40)
- `ALL_MODELS` (line 44)
- `EXCLUDE_TYPES` (line 101)
- `compute_entity_refs()` function definition (lines 171-201) with import
- `normalize_embeddings()` function definition (lines 204-211) with import
- Co-occurrence building (lines 512-529)
- `MIN_EDGE_WEIGHT` (line 477)

Keep: all model comparison logic, ground truth evaluation, disambiguation tests, and the reporting functions -- those are unique.

**Step 2: Commit**

```bash
git add notebooks/06_compare_models.py
git commit -m "refactor: 06_compare_models uses common modules"
```

---

### Task 14: Update 07_export_scores.py and 01_classify_tags.py to use common modules

**Files:**
- Modify: `notebooks/07_export_scores.py`
- Modify: `notebooks/01_classify_tags.py`

**Step 1: Update 07_export_scores.py**

Replace:
- Path setup (lines 36-39)
- `ALL_MODELS` (line 43)
- `EXCLUDE_TYPES` (line 117)
- Embedding loading (use `load_embeddings()`)
- Normalization (use `normalize_embeddings()`)

Keep: K-means multi-prototype logic, calibration stats, specificity computation -- those are this script's core.

**Step 2: Update 01_classify_tags.py**

This script has minimal duplication (just `wob_path` on line 19). Replace with import from `common.paths`.

**Step 3: Run Python tests to verify pipeline**

Run: `cd /Users/varunr/projects/cosmere-graph && python -m pytest tests/test_export_scores.py -v --timeout=300 2>&1 | tail -20`
Expected: Tests pass (or skip if embedding data unavailable). The pipeline output should be identical.

**Step 4: Commit**

```bash
git add notebooks/07_export_scores.py notebooks/01_classify_tags.py
git commit -m "refactor: remaining pipeline scripts use common modules"
```

---

## Phase 3: Frontend -- CSS Extraction

### Task 15: Extract CSS into separate files

**Files:**
- Create: `css/theme.css`
- Create: `css/layout.css`
- Create: `css/components.css`
- Create: `css/graph.css`
- Create: `css/animations.css`
- Modify: `index.html` -- replace `<style>...</style>` with `<link>` tags

**Step 1: Create css/ directory and split the CSS**

From `index.html` lines 10-1463, split by section comments:

- **`css/theme.css`**: Lines 11-51 (Variables + body + reset). Also include scrollbar styles (lines 575-583) and Google Fonts link will stay in HTML `<head>`.
- **`css/animations.css`**: Lines 53-128 (Loading screen + keyframes).
- **`css/graph.css`**: Lines 129-186 (Particle canvas + graph container + SVG styles) and lines 523-574 (Tooltip + info bar).
- **`css/layout.css`**: Lines 187-291 (Header, search) + lines 292-430 (Side panel) + lines 1020-1033 (Backdrop overlay) + tablet media query (lines 1034-1172) + phone media query (lines 1456-1463).
- **`css/components.css`**: Lines 251-291 (Filters) + lines 431-522 (Entries) + lines 584-1010 (Embedding controls, tuning panel, edge layer toggle, review panel) + lines 1173-1455 (Guide modal).

**Step 2: Update index.html**

Replace the entire `<style>...</style>` block (lines 10-1463) with:
```html
<link rel="stylesheet" href="css/theme.css">
<link rel="stylesheet" href="css/animations.css">
<link rel="stylesheet" href="css/graph.css">
<link rel="stylesheet" href="css/layout.css">
<link rel="stylesheet" href="css/components.css">
```

**Step 3: Run tests**

Run: `npx playwright test tests/test_controls.spec.js --reporter=list`
Expected: Tests pass. Visual appearance unchanged.

**Step 4: Commit**

```bash
git add css/ index.html
git commit -m "refactor: extract CSS into separate files"
```

---

## Phase 4: Frontend -- JS Module Extraction

### Task 16: Create js/constants.js and js/state.js

**Files:**
- Create: `js/constants.js`
- Create: `js/state.js`

**Step 1: Create constants.js**

Extract from `index.html` lines 1762-1800:

```js
// js/constants.js
export const GEM_COLORS = {
  character: '#2E5BA8',
  world:     '#2D8B57',
  magic:     '#7B4BAA',
  shard:     '#C49A2A',
  concept:   '#B03A3A',
};

export const GEM_GLOW = {
  character: '#4A8BDF',
  world:     '#3FBF7F',
  magic:     '#A66ED8',
  shard:     '#E8C44A',
  concept:   '#D45A5A',
};

export const GEM_HIGHLIGHT = {
  character: '#6AADFF',
  world:     '#5FE8A0',
  magic:     '#C490F0',
  shard:     '#F0D870',
  concept:   '#E87070',
};

export const GEM_NAMES = {
  character: 'Sapphire',
  world:     'Emerald',
  magic:     'Amethyst',
  shard:     'Heliodor',
  concept:   'Ruby',
};

export const TYPE_LABELS = {
  character: 'Characters',
  world:     'Worlds',
  magic:     'Magic Systems',
  shard:     'Shards',
  concept:   'Concepts',
};
```

**Step 2: Create state.js**

Extract from `index.html` lines 1952-1958 and lines 2886-2888, 3028-3030, 3167:

```js
// js/state.js -- shared mutable state for the application
import { GEM_COLORS } from './constants.js';

export let graph = null;
export let entries = null;
export let similarity = {};
export let simulation = null;
export let focusedNode = null;
export let activeFilters = new Set(Object.keys(GEM_COLORS));

// Embedding-related state
export let scoresData = null;
export let explicitTagsByEntry = {};
export let baselineConnected = {};

// Hypothesis layer state
export let implicitLinkGroup = null;
export let implicitLinks = null;
export let _originalTick = null;

// Review state
export let reviewState = {};

// Mobile backdrop refs
export let _panelBackdrop = null;
export let _reviewBackdrop = null;

// Setter functions (since ES module exports are read-only bindings for importers)
export function setGraph(val) { graph = val; }
export function setEntries(val) { entries = val; }
export function setSimilarity(val) { similarity = val; }
export function setSimulation(val) { simulation = val; }
export function setFocusedNode(val) { focusedNode = val; }
export function setScoresData(val) { scoresData = val; }
export function setImplicitLinkGroup(val) { implicitLinkGroup = val; }
export function setImplicitLinks(val) { implicitLinks = val; }
export function set_originalTick(val) { _originalTick = val; }
export function set_panelBackdrop(val) { _panelBackdrop = val; }
export function set_reviewBackdrop(val) { _reviewBackdrop = val; }
```

**Step 3: Commit**

```bash
git add js/constants.js js/state.js
git commit -m "refactor: create JS constants and state modules"
```

---

### Task 17: Create js/particles.js and js/graph.js

**Files:**
- Create: `js/particles.js`
- Create: `js/graph.js`

**Step 1: Create particles.js**

Extract the `StormParticles` class (lines 1874-1937) and `gemPath` function (lines 1941-1950):

```js
// js/particles.js
export class StormParticles { /* ... exact copy of lines 1874-1937 ... */ }
export function gemPath(r, facets) { /* ... exact copy of lines 1941-1950 ... */ }
```

**Step 2: Create graph.js**

Extract `buildGraph()` function (lines 2023-2202). It needs imports:
```js
import { GEM_COLORS, GEM_GLOW, GEM_HIGHLIGHT, GEM_NAMES } from './constants.js';
import * as state from './state.js';
import { gemPath } from './particles.js';
```

Replace all bare references to globals (`graph`, `simulation`, `focusedNode`, etc.) with `state.graph`, `state.simulation`, etc. Replace assignments like `simulation = d3.forceSimulation(...)` with `state.setSimulation(d3.forceSimulation(...))`.

Store D3 selections on `window._nodes`, `window._links`, `window._svg`, `window._zoom`, `window._container`, `window._sizeScale` as they already are in the original code -- these are used as shared state across modules.

Export: `export function buildGraph() { ... }`

**Step 3: Commit**

```bash
git add js/particles.js js/graph.js
git commit -m "refactor: extract particles and graph modules"
```

---

### Task 18: Create js/panel.js, js/filters.js, js/search.js

**Files:**
- Create: `js/panel.js`
- Create: `js/filters.js`
- Create: `js/search.js`

**Step 1: Create panel.js**

Extract from lines 2204-2733: `getNeighbors()`, `focusNode()`, `unfocus()`, `setupPanel()`, `showPanel()`, `closePanel()`, and all the panel content building functions.

Imports needed:
```js
import { GEM_COLORS, GEM_GLOW, GEM_HIGHLIGHT, GEM_NAMES, TYPE_LABELS } from './constants.js';
import * as state from './state.js';
```

Export: `getNeighbors`, `focusNode`, `unfocus`, `setupPanel`, `showPanel`, `closePanel`

**Step 2: Create filters.js**

Extract from lines 2734-2786: `buildFilters()`, `toggleFilter()`, `applyFilters()`.

```js
import { GEM_COLORS, TYPE_LABELS } from './constants.js';
import * as state from './state.js';
```

Export: `buildFilters`, `applyFilters`

**Step 3: Create search.js**

Extract from lines 2788-2882: `setupSearch()` and the keyboard shortcut listener.

```js
import { GEM_COLORS, GEM_GLOW, GEM_NAMES } from './constants.js';
import * as state from './state.js';
import { focusNode, unfocus } from './panel.js';
```

Export: `setupSearch`

**Step 4: Commit**

```bash
git add js/panel.js js/filters.js js/search.js
git commit -m "refactor: extract panel, filters, and search modules"
```

---

### Task 19: Create js/embeddings.js, js/hypothesis.js, js/review.js

**Files:**
- Create: `js/embeddings.js`
- Create: `js/hypothesis.js`
- Create: `js/review.js`

**Step 1: Create embeddings.js**

Extract from lines 2884-3024: `buildExplicitTagsByEntry()`, `buildBaselineConnected()`, `setupEmbeddingControls()`, `loadScoresAndCompute()`.

```js
import * as state from './state.js';
import { renderImplicitEdges } from './hypothesis.js';
import { populateReviewPanel } from './review.js';
```

Export: `buildExplicitTagsByEntry`, `buildBaselineConnected`, `setupEmbeddingControls`

**Step 2: Create hypothesis.js**

Extract from lines 3026-3163: `renderImplicitEdges()`, `setupEdgeLayerToggle()`, `applyEdgeLayerFilter()`.

```js
import * as state from './state.js';
```

Export: `renderImplicitEdges`, `setupEdgeLayerToggle`, `applyEdgeLayerFilter`

**Step 3: Create review.js**

Extract from lines 3165-3402: `setupReviewPanel()`, `populateReviewPanel()`, `saveReviews()`, `loadReviews()`, and all review UI logic.

```js
import * as state from './state.js';
```

Export: `setupReviewPanel`, `populateReviewPanel`

**Step 4: Commit**

```bash
git add js/embeddings.js js/hypothesis.js js/review.js
git commit -m "refactor: extract embeddings, hypothesis, and review modules"
```

---

### Task 20: Create js/mobile.js and js/guide.js

**Files:**
- Create: `js/mobile.js`
- Create: `js/guide.js`

**Step 1: Create mobile.js**

Extract from lines 1802-1840: `isMobile()`, `createBackdrop()`, `showBackdrop()`, `hideBackdrop()`, `setupBackdrops()`.

```js
import * as state from './state.js';
import { unfocus } from './panel.js';
```

Export: `isMobile`, `showBackdrop`, `hideBackdrop`, `setupBackdrops`

**Step 2: Create guide.js**

Extract from lines 1842-1870: `setupGuide()`.

Export: `setupGuide`

**Step 3: Commit**

```bash
git add js/mobile.js js/guide.js
git commit -m "refactor: extract mobile and guide modules"
```

---

### Task 21: Create js/app.js and update index.html

This is the critical integration step. Create the entry point module and strip all JS from index.html.

**Files:**
- Create: `js/app.js`
- Modify: `index.html` -- remove the entire `<script>...</script>` block (lines 1759-3406), replace with a single `<script type="module" src="js/app.js"></script>`

**Step 1: Create app.js**

```js
// js/app.js -- Application entry point
import * as state from './state.js';
import { StormParticles } from './particles.js';
import { buildGraph } from './graph.js';
import { buildFilters } from './filters.js';
import { setupSearch } from './search.js';
import { setupPanel } from './panel.js';
import { buildExplicitTagsByEntry, buildBaselineConnected, setupEmbeddingControls } from './embeddings.js';
import { setupEdgeLayerToggle } from './hypothesis.js';
import { setupReviewPanel } from './review.js';
import { setupBackdrops } from './mobile.js';
import { setupGuide } from './guide.js';

async function init() {
  const startTime = Date.now();

  // Start particles immediately
  const storm = new StormParticles(document.getElementById('particles'));
  storm.init(70);

  // Load data
  const basePath = window.location.pathname.replace(/\/[^\/]*$/, '');
  const [graphResp, entriesResp, similarityResp] = await Promise.all([
    fetch(basePath + '/data/graph.json').then(r => r.json()),
    fetch(basePath + '/data/entries.json').then(r => r.json()),
    fetch(basePath + '/data/similarity.json').then(r => r.json()).catch(() => ({})),
  ]);
  state.setGraph(graphResp);
  state.setEntries(entriesResp);
  state.setSimilarity(similarityResp);

  // Filter book nodes
  state.graph.nodes = state.graph.nodes.filter(n => n.type !== 'book');
  const nodeIds = new Set(state.graph.nodes.map(n => n.id));
  state.graph.edges = state.graph.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));

  document.getElementById('node-count').textContent = state.graph.nodes.length;
  document.getElementById('edge-count').textContent = state.graph.edges.length;

  // Build graph while loading screen is visible
  buildGraph();
  buildFilters();
  setupSearch();
  setupPanel();
  buildExplicitTagsByEntry();
  buildBaselineConnected();
  setupEmbeddingControls();
  setupEdgeLayerToggle();
  setupReviewPanel();
  setupBackdrops();
  setupGuide();

  // Ensure minimum 2.2s loading for animation
  const elapsed = Date.now() - startTime;
  if (elapsed < 2200) {
    await new Promise(r => setTimeout(r, 2200 - elapsed));
  }

  // Transition out
  const loading = document.getElementById('loading');
  loading.classList.add('fade-out');
  await new Promise(r => setTimeout(r, 600));
  loading.style.display = 'none';

  // Fade in graph and UI
  document.getElementById('graph-container').style.opacity = '1';
  document.getElementById('header').style.opacity = '1';
  document.getElementById('filters').style.opacity = '1';
  document.getElementById('info').style.opacity = '1';
  document.getElementById('embedding-controls-bar').style.opacity = '1';
}

init();
```

**Step 2: Update index.html**

Remove lines 1759-3406 (the entire `<script>` block). Replace with:
```html
<script src="https://d3js.org/d3.v7.min.js"></script>
<script type="module" src="js/app.js"></script>
```

Note: `tagging-engine.js` will need to be imported by `js/embeddings.js` (the module that calls `computeImplicitTags`). Since `tagging-engine.js` currently uses global function declarations, it needs either:
- A `<script src="js/tagging-engine.js"></script>` kept before the module script (globals still work), OR
- Minor updates to add `export` to its public functions

The simplest approach: keep the tagging-engine as a global script tag and access its functions via `window.computeImplicitTags` from within ES modules. This avoids modifying the tagging-engine (which has its own test suite).

```html
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="js/tagging-engine.js"></script>
<script type="module" src="js/app.js"></script>
```

**Step 3: Run ALL Playwright tests**

Run: `npx playwright test tests/ --reporter=list`
Expected: All tests pass. The app should work identically to before.

**Step 4: Commit**

```bash
git add js/app.js index.html
git commit -m "refactor: create app.js entry point, strip JS from index.html"
```

---

## Phase 5: Final Verification and Cleanup

### Task 22: Full test suite verification

**Step 1: Run all Playwright tests**

Run: `npx playwright test tests/ --reporter=list`
Expected: All tests pass.

**Step 2: Run Python tests (if embedding data available)**

Run: `python -m pytest tests/test_export_scores.py -v`
Expected: Tests pass or skip gracefully.

**Step 3: Manual spot check**

Run: `python3 -m http.server 8080 --bind 127.0.0.1`
Open http://127.0.0.1:8080/index.html in a browser. Verify:
- Loading animation plays
- Graph renders with all entity types
- Search works (type "kaladin")
- Filters toggle correctly
- Side panel opens on node click
- Embedding controls Apply button works
- Worldhopper's Guide modal opens

**Step 4: Commit**

No code changes -- this is verification only. If issues are found, fix and commit.

---

### Task 23: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Update the architecture section to reflect the new file structure. Add the new `make` commands. Remove references to the monolithic `index.html` structure.

**Step 1: Update CLAUDE.md**

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for refactored structure"
```
