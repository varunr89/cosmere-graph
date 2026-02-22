// js/filters.js -- Type filter buttons and filter logic

import { GEM_COLORS, TYPE_LABELS } from './constants.js';
import * as state from './state.js';

export function buildFilters() {
  var container = document.getElementById('filters');
  Object.keys(TYPE_LABELS).forEach(function(type) {
    var btn = document.createElement('button');
    btn.className = 'filter-btn';
    btn.dataset.type = type;

    var gem = document.createElement('span');
    gem.className = 'filter-gem';
    gem.style.background = GEM_COLORS[type];
    btn.appendChild(gem);

    var label = document.createTextNode(TYPE_LABELS[type]);
    btn.appendChild(label);

    btn.addEventListener('mouseenter', function() {
      if (!btn.classList.contains('off')) {
        btn.style.borderColor = GEM_COLORS[type];
      }
    });
    btn.addEventListener('mouseleave', function() {
      btn.style.borderColor = '';
    });
    btn.addEventListener('click', function() { toggleFilter(type, btn); });
    container.appendChild(btn);
  });
}

function toggleFilter(type, btn) {
  if (state.activeFilters.has(type)) {
    state.activeFilters.delete(type);
    btn.classList.add('off');
  } else {
    state.activeFilters.add(type);
    btn.classList.remove('off');
  }
  applyFilters();
}

export function applyFilters() {
  window._nodes.select('.gem-node')
    .transition().duration(200)
    .attr('fill-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.85 : 0; })
    .attr('stroke-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.3 : 0; });
  window._nodes.select('.gem-glow')
    .transition().duration(200)
    .attr('fill-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.1 : 0; });
  window._nodes.select('.gem-label')
    .transition().duration(200)
    .attr('fill-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.75 : 0; });

  // Hide edges connected to filtered-out node types
  function edgeVisible(d) {
    var srcType = typeof d.source === 'object' ? d.source.type : null;
    var tgtType = typeof d.target === 'object' ? d.target.type : null;
    return state.activeFilters.has(srcType) && state.activeFilters.has(tgtType);
  }
  if (window._links) {
    window._links
      .transition().duration(200)
      .attr('stroke-opacity', function(d) { return edgeVisible(d) ? 1 : 0; });
  }
  if (state.implicitLinks) {
    state.implicitLinks
      .transition().duration(200)
      .attr('stroke-opacity', function(d) {
        if (!edgeVisible(d)) return 0;
        return d.type === 'mixed' ? 0.55 : 0.4;
      });
  }
}
