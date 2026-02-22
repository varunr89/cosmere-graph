// js/graph.js -- D3 force simulation graph setup

import { GEM_COLORS, GEM_GLOW, GEM_HIGHLIGHT, GEM_NAMES } from './constants.js';
import * as state from './state.js';
import { gemPath } from './particles.js';
import { focusNode, unfocus } from './panel.js';
import { isMobile } from './mobile.js';

export function buildGraph() {
  var svg = d3.select('#graph-svg');
  var width = window.innerWidth;
  var height = window.innerHeight;
  var defs = svg.append('defs');

  // Soft blur filter for glow circles
  var blurFilter = defs.append('filter')
    .attr('id', 'soft-blur')
    .attr('x', '-100%').attr('y', '-100%')
    .attr('width', '300%').attr('height', '300%');
  blurFilter.append('feGaussianBlur')
    .attr('stdDeviation', '5');

  // Radial gradients for gemstone fills
  Object.keys(GEM_COLORS).forEach(function(type) {
    var grad = defs.append('radialGradient')
      .attr('id', 'gem-fill-' + type)
      .attr('cx', '35%').attr('cy', '30%').attr('r', '65%');
    grad.append('stop').attr('offset', '0%').attr('stop-color', GEM_HIGHLIGHT[type]);
    grad.append('stop').attr('offset', '100%').attr('stop-color', GEM_COLORS[type]);
  });

  // Scales
  var sizeScale = d3.scaleSqrt()
    .domain([1, d3.max(state.graph.nodes, function(d) { return d.entryCount; })])
    .range([4, 24]);

  var edgeScale = d3.scaleLinear()
    .domain([2, d3.max(state.graph.edges, function(d) { return d.weight; })])
    .range([0.3, 2.5]);

  // Zoom
  var zoom = d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', function(event) { container.attr('transform', event.transform); });
  svg.call(zoom);
  svg.on('click', function(event) {
    if (event.target === svg.node()) unfocus();
  });

  var container = svg.append('g');

  // Edges
  var linkGroup = container.append('g').attr('class', 'links');
  var links = linkGroup.selectAll('line')
    .data(state.graph.edges)
    .join('line')
    .attr('stroke', 'rgba(200,223,255,0.06)')
    .attr('stroke-width', function(d) { return edgeScale(d.weight); })
    .attr('stroke-opacity', 1);

  // Nodes
  var nodeGroup = container.append('g').attr('class', 'nodes');
  var nodes = nodeGroup.selectAll('g')
    .data(state.graph.nodes)
    .join('g')
    .attr('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragStart)
      .on('drag', dragging)
      .on('end', dragEnd));

  // Glow circle (behind gemstone)
  nodes.append('circle')
    .attr('class', 'gem-glow')
    .attr('r', function(d) { return sizeScale(d.entryCount) * 1.8; })
    .attr('fill', function(d) { return GEM_GLOW[d.type]; })
    .attr('fill-opacity', 0.1)
    .attr('filter', 'url(#soft-blur)')
    .attr('pointer-events', 'none');

  // Gemstone node
  nodes.append('path')
    .attr('class', 'gem-node')
    .attr('d', function(d) { return gemPath(sizeScale(d.entryCount), d.entryCount > 80 ? 8 : 6); })
    .attr('fill', function(d) { return 'url(#gem-fill-' + d.type + ')'; })
    .attr('fill-opacity', 0.85)
    .attr('stroke', function(d) { return GEM_GLOW[d.type]; })
    .attr('stroke-width', 0.8)
    .attr('stroke-opacity', 0.3);

  // Labels
  nodes.filter(function(d) { return d.entryCount >= 30; })
    .append('text')
    .attr('class', 'gem-label')
    .text(function(d) { return d.label; })
    .attr('text-anchor', 'middle')
    .attr('dy', function(d) { return sizeScale(d.entryCount) + 14; })
    .attr('font-size', function(d) { return Math.min(11, 7 + d.entryCount / 50); })
    .attr('fill', '#C8DFFF')
    .attr('fill-opacity', 0.75)
    .attr('pointer-events', 'none')
    .attr('font-family', 'system-ui, sans-serif');

  // Tooltip
  var tooltip = d3.select('#tooltip');
  nodes.on('mouseenter', function(event, d) {
    var nameEl = document.createElement('div');
    nameEl.className = 'tt-name';
    nameEl.style.color = GEM_GLOW[d.type];
    nameEl.textContent = d.label;

    var metaEl = document.createElement('div');
    metaEl.className = 'tt-meta';
    metaEl.textContent = GEM_NAMES[d.type] + ' \u00B7 ' + d.entryCount + ' entries';

    var ttContainer = tooltip.node();
    ttContainer.textContent = '';
    ttContainer.appendChild(nameEl);
    ttContainer.appendChild(metaEl);
    tooltip.style('display', 'block');
  })
  .on('mousemove', function(event) {
    tooltip.style('left', (event.clientX + 14) + 'px')
      .style('top', (event.clientY - 10) + 'px');
  })
  .on('mouseleave', function() { tooltip.style('display', 'none'); });

  // Mobile tooltip: show briefly on touch
  nodes.on('touchstart', function(event, d) {
    if (!isMobile()) return;
    var touch = event.touches[0];
    var nameEl = document.createElement('div');
    nameEl.className = 'tt-name';
    nameEl.style.color = GEM_GLOW[d.type];
    nameEl.textContent = d.label;
    var metaEl = document.createElement('div');
    metaEl.className = 'tt-meta';
    metaEl.textContent = GEM_NAMES[d.type] + ' \u00B7 ' + d.entryCount + ' entries';
    var ttNode = tooltip.node();
    ttNode.textContent = '';
    ttNode.appendChild(nameEl);
    ttNode.appendChild(metaEl);
    tooltip.style('display', 'block')
      .style('left', (touch.clientX + 14) + 'px')
      .style('top', (touch.clientY - 10) + 'px');
    clearTimeout(window._tooltipTimer);
    window._tooltipTimer = setTimeout(function() {
      tooltip.style('display', 'none');
    }, 2000);
  }, { passive: true });

  // Click to focus
  nodes.on('click', function(event, d) {
    event.stopPropagation();
    focusNode(d.id);
  });

  // Simulation
  state.setSimulation(
    d3.forceSimulation(state.graph.nodes)
      .force('link', d3.forceLink(state.graph.edges).id(function(d) { return d.id; }).distance(80).strength(function(d) { return Math.min(0.3, d.weight / 50); }))
      .force('charge', d3.forceManyBody().strength(-120).distanceMax(400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(function(d) { return sizeScale(d.entryCount) + 4; }))
      .on('tick', function() {
        links
          .attr('x1', function(d) { return d.source.x; }).attr('y1', function(d) { return d.source.y; })
          .attr('x2', function(d) { return d.target.x; }).attr('y2', function(d) { return d.target.y; });
        nodes.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
      })
  );

  // Store refs
  window._nodes = nodes;
  window._links = links;
  window._sizeScale = sizeScale;
  window._zoom = zoom;
  window._svg = svg;
  window._container = container;

  function dragStart(event, d) {
    if (!event.active) state.simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragging(event, d) { d.fx = event.x; d.fy = event.y; }
  function dragEnd(event, d) {
    if (!event.active) state.simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }
}
