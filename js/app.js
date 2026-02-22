// js/app.js -- Entry point: init and orchestration

import * as state from './state.js';
import { StormParticles } from './particles.js';
import { buildGraph } from './graph.js';
import { setupPanel } from './panel.js';
import { buildFilters } from './filters.js';
import { setupSearch } from './search.js';
import { buildExplicitTagsByEntry, buildBaselineConnected, setupEmbeddingControls } from './embeddings.js';
import { setupEdgeLayerToggle } from './hypothesis.js';
import { setupReviewPanel } from './review.js';
import { setupBackdrops } from './mobile.js';
import { setupGuide } from './guide.js';

async function init() {
  var startTime = Date.now();

  // Start particles immediately
  var storm = new StormParticles(document.getElementById('particles'));
  storm.init(70);

  // Load data
  var basePath = window.location.pathname.replace(/\/[^\/]*$/, '');
  var responses = await Promise.all([
    fetch(basePath + '/data/graph.json').then(function(r) { return r.json(); }),
    fetch(basePath + '/data/entries.json').then(function(r) { return r.json(); }),
    fetch(basePath + '/data/similarity.json').then(function(r) { return r.json(); }).catch(function() { return {}; }),
  ]);
  state.setGraph(responses[0]);
  state.setEntries(responses[1]);
  state.setSimilarity(responses[2]);

  // Filter book nodes
  state.graph.nodes = state.graph.nodes.filter(function(n) { return n.type !== 'book'; });
  var nodeIds = new Set(state.graph.nodes.map(function(n) { return n.id; }));
  state.graph.edges = state.graph.edges.filter(function(e) { return nodeIds.has(e.source) && nodeIds.has(e.target); });

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

  // Expose globals needed by Playwright tests
  window.reviewState = state.reviewState;

  // Ensure minimum 2.2s loading for animation
  var elapsed = Date.now() - startTime;
  if (elapsed < 2200) {
    await new Promise(function(r) { return setTimeout(r, 2200 - elapsed); });
  }

  // Transition out
  var loading = document.getElementById('loading');
  loading.classList.add('fade-out');
  await new Promise(function(r) { return setTimeout(r, 600); });
  loading.style.display = 'none';

  // Fade in graph and UI
  document.getElementById('graph-container').style.opacity = '1';
  document.getElementById('header').style.opacity = '1';
  document.getElementById('filters').style.opacity = '1';
  document.getElementById('info').style.opacity = '1';
  document.getElementById('embedding-controls-bar').style.opacity = '1';
}

init();
