// js/embeddings.js -- Embedding controls, score loading, and computation

import * as state from './state.js';
import { renderImplicitEdges } from './hypothesis.js';
import { populateReviewPanel } from './review.js';
import { applyFilters } from './filters.js';

export function buildExplicitTagsByEntry() {
  var eids = Object.keys(state.entries);
  for (var i = 0; i < eids.length; i++) {
    var eid = eids[i];
    var entry = state.entries[eid];
    if (entry.tags && entry.tags.length > 0) {
      state.explicitTagsByEntry[eid] = entry.tags;
    }
  }
}

export function buildBaselineConnected() {
  // Entries that appear in at least one edge in graph.json
  for (var i = 0; i < state.graph.edges.length; i++) {
    var e = state.graph.edges[i];
    var src = typeof e.source === 'object' ? e.source.id : e.source;
    var tgt = typeof e.target === 'object' ? e.target.id : e.target;
    state.baselineConnected[src] = true;
    state.baselineConnected[tgt] = true;
  }
}

export function setupEmbeddingControls() {
  var tuningToggle = document.getElementById('tuning-toggle-btn');
  var tuningPanel = document.getElementById('tuning-panel');
  var applyBtn = document.getElementById('apply-embeddings-btn');
  var statsEl = document.getElementById('embedding-stats');

  // Slider value labels: wire up input events
  var sliders = [
    { id: 'slider-calibration-percentile', valueId: 'value-calibration-percentile', format: function(v) { return String(parseInt(v, 10)); } },
    { id: 'slider-min-specificity',        valueId: 'value-min-specificity',        format: function(v) { return parseFloat(v).toFixed(1); } },
    { id: 'slider-confidence-margin',      valueId: 'value-confidence-margin',      format: function(v) { return parseFloat(v).toFixed(2); } },
    { id: 'slider-min-edge-weight',        valueId: 'value-min-edge-weight',        format: function(v) { return String(parseInt(v, 10)); } }
  ];

  for (var i = 0; i < sliders.length; i++) {
    (function(s) {
      var slider = document.getElementById(s.id);
      var valueEl = document.getElementById(s.valueId);
      slider.addEventListener('input', function() {
        valueEl.textContent = s.format(slider.value);
      });
    })(sliders[i]);
  }

  // Toggle tuning panel
  tuningToggle.addEventListener('click', function() {
    var isOpen = tuningPanel.classList.contains('open');
    if (isOpen) {
      tuningPanel.classList.remove('open');
      tuningToggle.textContent = 'Tune';
    } else {
      tuningPanel.classList.add('open');
      tuningToggle.textContent = 'Hide';
    }
  });

  // Apply button
  applyBtn.addEventListener('click', function() {
    applyBtn.disabled = true;
    applyBtn.textContent = 'Computing...';
    statsEl.textContent = 'Loading scores...';

    // Use setTimeout to allow UI to update before heavy computation
    setTimeout(function() {
      loadScoresAndCompute(applyBtn, statsEl);
    }, 50);
  });
}

function loadScoresAndCompute(applyBtn, statsEl) {
  var basePath = window.location.pathname.replace(/\/[^\/]*$/, '');

  function runCompute() {
    // Read settings from sliders
    var settings = {
      calibrationPercentile: parseInt(document.getElementById('slider-calibration-percentile').value, 10),
      minSpecificity: parseFloat(document.getElementById('slider-min-specificity').value),
      confidenceMargin: parseFloat(document.getElementById('slider-confidence-margin').value),
      mustBridge: document.getElementById('checkbox-must-bridge').checked,
      minEdgeWeight: parseInt(document.getElementById('slider-min-edge-weight').value, 10)
    };

    try {
      var result = window.computeImplicitTags(state.scoresData, state.explicitTagsByEntry, state.baselineConnected, settings);
      var s = result.stats;

      statsEl.textContent = s.totalTags + ' tags, ' +
        s.entitiesConsidered + ' entities, ' +
        s.totalEdges + ' edges (' +
        s.implicitEdges + ' implicit, ' +
        s.mixedEdges + ' mixed)';

      // Store result globally for tests and review panel
      window._lastImplicitResult = result;

      // Render hypothesis layer (implicit edges) and populate review panel
      renderImplicitEdges(result);
      applyFilters();
      populateReviewPanel(result);

      // Show edge layer toggle and review button
      document.getElementById('edge-layer-toggle').classList.add('visible');
      document.getElementById('review-toggle-btn').classList.add('visible');

    } catch (err) {
      statsEl.textContent = 'Error: ' + err.message;
    }

    applyBtn.disabled = false;
    applyBtn.textContent = 'Apply';
  }

  // Load scores.json if not already cached
  if (state.scoresData) {
    runCompute();
  } else {
    fetch(basePath + '/data/scores.json')
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(data) {
        state.setScoresData(data);
        statsEl.textContent = 'Computing...';
        // Another setTimeout so UI updates before heavy computation
        setTimeout(runCompute, 50);
      })
      .catch(function(err) {
        statsEl.textContent = 'Failed to load scores: ' + err.message;
        applyBtn.disabled = false;
        applyBtn.textContent = 'Apply';
      });
  }
}
