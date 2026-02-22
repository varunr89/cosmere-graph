// js/hypothesis.js -- Implicit edge rendering and edge layer toggle

import * as state from './state.js';

export function renderImplicitEdges(result) {
  var container = window._container;
  if (!container) return;

  // Remove previous implicit edges if any
  if (state.implicitLinkGroup) {
    state.implicitLinkGroup.remove();
  }

  // Filter edges to only implicit and mixed types
  var implicitEdges = [];
  for (var i = 0; i < result.edges.length; i++) {
    var e = result.edges[i];
    if (e.type === 'implicit' || e.type === 'mixed') {
      implicitEdges.push(e);
    }
  }

  if (implicitEdges.length === 0) return;

  // Build a map from entity id to simulation node for position lookup
  var nodeMap = {};
  for (var i = 0; i < state.graph.nodes.length; i++) {
    nodeMap[state.graph.nodes[i].id] = state.graph.nodes[i];
  }

  // Filter edges to only those whose source and target exist in the graph
  var validEdges = [];
  for (var i = 0; i < implicitEdges.length; i++) {
    var e = implicitEdges[i];
    if (nodeMap[e.source] && nodeMap[e.target]) {
      validEdges.push({
        source: nodeMap[e.source],
        target: nodeMap[e.target],
        weight: e.weight,
        type: e.type,
        entryIds: e.entryIds
      });
    }
  }

  // Insert implicit edge group before the nodes group (so edges are behind nodes)
  var nodesGroup = container.select('g.nodes');
  state.setImplicitLinkGroup(
    container.insert('g', function() { return nodesGroup.node(); })
      .attr('class', 'implicit-links')
  );

  state.setImplicitLinks(
    state.implicitLinkGroup.selectAll('line')
      .data(validEdges)
      .join('line')
      .attr('class', 'implicit-edge')
      .attr('stroke', 'var(--gem-heliodor-glow)')
      .attr('stroke-width', function(d) { return Math.max(0.5, Math.min(2, d.weight / 3)); })
      .attr('stroke-opacity', function(d) { return d.type === 'mixed' ? 0.55 : 0.4; })
      .attr('stroke-dasharray', '6,4')
      .attr('x1', function(d) { return d.source.x || 0; })
      .attr('y1', function(d) { return d.source.y || 0; })
      .attr('x2', function(d) { return d.target.x || 0; })
      .attr('y2', function(d) { return d.target.y || 0; })
  );

  // Add tooltip behavior for implicit edges
  var tooltip = d3.select('#tooltip');
  state.implicitLinks
    .style('pointer-events', 'stroke')
    .on('mouseenter', function(event, d) {
      var srcLabel = d.source.label || d.source.id;
      var tgtLabel = d.target.label || d.target.id;

      var nameEl = document.createElement('div');
      nameEl.className = 'tt-name';
      nameEl.style.color = 'var(--gem-heliodor-glow)';
      nameEl.textContent = srcLabel + ' - ' + tgtLabel;

      var metaEl = document.createElement('div');
      metaEl.className = 'tt-meta';
      metaEl.textContent = d.type + ' edge / weight ' + d.weight + ' / ' + d.entryIds.length + ' entries';

      var ttNode = tooltip.node();
      ttNode.textContent = '';
      ttNode.appendChild(nameEl);
      ttNode.appendChild(metaEl);
      tooltip.style('display', 'block')
        .style('pointer-events', 'none');
    })
    .on('mousemove', function(event) {
      tooltip.style('left', (event.clientX + 14) + 'px')
        .style('top', (event.clientY - 10) + 'px');
    })
    .on('mouseleave', function() {
      tooltip.style('display', 'none');
    });

  // Store original tick handler once, then extend it with implicit edge updates
  if (!state._originalTick) {
    state.set_originalTick(state.simulation.on('tick'));
  }
  state.simulation.on('tick', function() {
    state._originalTick();

    // Update implicit edges
    if (state.implicitLinks) {
      state.implicitLinks
        .attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });
    }
  });

  // Restart simulation with low alpha so it settles without big jumps
  state.simulation.alpha(0.1).restart();
}

export function setupEdgeLayerToggle() {
  var radios = document.querySelectorAll('input[name="edge-layer"]');
  for (var i = 0; i < radios.length; i++) {
    radios[i].addEventListener('change', function() {
      applyEdgeLayerFilter(this.value);
    });
  }
}

export function applyEdgeLayerFilter(mode) {
  // mode: 'explicit', 'both', 'implicit'
  if (window._links) {
    window._links.attr('visibility', (mode === 'implicit') ? 'hidden' : 'visible');
  }
  if (state.implicitLinks) {
    state.implicitLinks.attr('visibility', (mode === 'explicit') ? 'hidden' : 'visible');
  }
}
