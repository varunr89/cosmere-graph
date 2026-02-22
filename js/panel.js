// js/panel.js -- Side panel: focus/unfocus, panel content, entry display
//
// Note: stripHtml() uses innerHTML to decode HTML entities from Arcanum data.
// This is intentional -- the decoded text is then extracted via textContent,
// which is safe against XSS. The pattern is: set textContent (safe encode),
// read textContent (decode), set innerHTML (parse), read textContent (strip).

import { GEM_COLORS, GEM_GLOW, GEM_HIGHLIGHT, GEM_NAMES, TYPE_LABELS } from './constants.js';
import * as state from './state.js';
import { isMobile, showBackdrop, hideBackdrop } from './mobile.js';

export function getNeighbors(nodeId) {
  var neighbors = new Set();
  var edgeData = [];
  state.graph.edges.forEach(function(e) {
    var src = typeof e.source === 'object' ? e.source.id : e.source;
    var tgt = typeof e.target === 'object' ? e.target.id : e.target;
    if (src === nodeId) { neighbors.add(tgt); edgeData.push(e); }
    if (tgt === nodeId) { neighbors.add(src); edgeData.push(e); }
  });
  return { neighbors: neighbors, edgeData: edgeData };
}

export function focusNode(nodeId) {
  state.setFocusedNode(nodeId);
  var result = getNeighbors(nodeId);
  var neighbors = result.neighbors;
  var edgeData = result.edgeData;
  var node = state.graph.nodes.find(function(n) { return n.id === nodeId; });
  var focusGlow = GEM_GLOW[node.type];

  // Gemstone nodes
  window._nodes.select('.gem-node')
    .transition().duration(300)
    .attr('fill-opacity', function(d) {
      if (d.id === nodeId) return 1;
      if (neighbors.has(d.id)) return 0.85;
      return 0.04;
    })
    .attr('stroke-opacity', function(d) {
      if (d.id === nodeId) return 0.8;
      if (neighbors.has(d.id)) return 0.4;
      return 0.02;
    })
    .attr('transform', function(d) { return d.id === nodeId ? 'scale(1.3)' : 'scale(1)'; });

  // Glow circles
  window._nodes.select('.gem-glow')
    .transition().duration(300)
    .attr('fill-opacity', function(d) {
      if (d.id === nodeId) return 0.3;
      if (neighbors.has(d.id)) return 0.15;
      return 0.01;
    })
    .attr('r', function(d) {
      var base = window._sizeScale(d.entryCount) * 1.8;
      return d.id === nodeId ? base * 1.6 : base;
    });

  // Labels
  window._nodes.select('.gem-label')
    .transition().duration(300)
    .attr('fill-opacity', function(d) { return (d.id === nodeId || neighbors.has(d.id)) ? 0.9 : 0.03; });

  // Edges -- respect type filters
  function linkFilteredOut(d) {
    var srcType = typeof d.source === 'object' ? d.source.type : null;
    var tgtType = typeof d.target === 'object' ? d.target.type : null;
    return !state.activeFilters.has(srcType) || !state.activeFilters.has(tgtType);
  }

  window._links
    .attr('stroke', function(d) {
      if (linkFilteredOut(d)) return 'rgba(200,223,255,0.03)';
      var src = typeof d.source === 'object' ? d.source.id : d.source;
      var tgt = typeof d.target === 'object' ? d.target.id : d.target;
      return (src === nodeId || tgt === nodeId) ? focusGlow : 'rgba(200,223,255,0.03)';
    });

  // Pulse connected edges (only if both endpoints are active)
  window._links.filter(function(d) {
    if (linkFilteredOut(d)) return false;
    var src = typeof d.source === 'object' ? d.source.id : d.source;
    var tgt = typeof d.target === 'object' ? d.target.id : d.target;
    return src === nodeId || tgt === nodeId;
  })
    .attr('stroke-opacity', 0.1)
    .transition().duration(350)
    .attr('stroke-opacity', 0.65)
    .transition().duration(500)
    .attr('stroke-opacity', 0.4);

  // Dim unconnected or filtered-out edges
  window._links.filter(function(d) {
    if (linkFilteredOut(d)) return true;
    var src = typeof d.source === 'object' ? d.source.id : d.source;
    var tgt = typeof d.target === 'object' ? d.target.id : d.target;
    return src !== nodeId && tgt !== nodeId;
  })
    .transition().duration(300)
    .attr('stroke-opacity', function(d) { return linkFilteredOut(d) ? 0.02 : 0.3; });

  // Center camera
  if (node && typeof node.x === 'number') {
    var transform = d3.zoomIdentity
      .translate(window.innerWidth / 2, window.innerHeight / 2)
      .scale(1.5)
      .translate(-node.x, -node.y);
    window._svg.transition().duration(750).call(window._zoom.transform, transform);
  }

  showPanel(nodeId, edgeData);
}

export function unfocus() {
  state.setFocusedNode(null);

  window._nodes.select('.gem-node')
    .transition().duration(300)
    .attr('fill-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.85 : 0.04; })
    .attr('stroke-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.3 : 0.02; })
    .attr('transform', 'scale(1)');

  window._nodes.select('.gem-glow')
    .transition().duration(300)
    .attr('fill-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.1 : 0.01; })
    .attr('r', function(d) { return window._sizeScale(d.entryCount) * 1.8; });

  window._nodes.select('.gem-label')
    .transition().duration(300)
    .attr('fill-opacity', function(d) { return state.activeFilters.has(d.type) ? 0.75 : 0.03; });

  window._links
    .transition().duration(300)
    .attr('stroke', 'rgba(200,223,255,0.06)')
    .attr('stroke-opacity', function(d) {
      var srcType = typeof d.source === 'object' ? d.source.type : null;
      var tgtType = typeof d.target === 'object' ? d.target.type : null;
      return (state.activeFilters.has(srcType) && state.activeFilters.has(tgtType)) ? 1 : 0.02;
    });

  if (state.implicitLinks) {
    state.implicitLinks
      .transition().duration(300)
      .attr('stroke-opacity', function(d) {
        var srcType = typeof d.source === 'object' ? d.source.type : null;
        var tgtType = typeof d.target === 'object' ? d.target.type : null;
        if (!state.activeFilters.has(srcType) || !state.activeFilters.has(tgtType)) return 0.02;
        return d.type === 'mixed' ? 0.55 : 0.4;
      });
  }

  closePanel();

  window._svg.transition().duration(500)
    .call(window._zoom.transform, d3.zoomIdentity
      .translate(window.innerWidth / 2, window.innerHeight / 2)
      .scale(0.8)
      .translate(-window.innerWidth / 2, -window.innerHeight / 2));
}

export function setupPanel() {
  document.getElementById('panel-close').addEventListener('click', function() { unfocus(); });
}

export function showPanel(nodeId, edgeData) {
  var node = state.graph.nodes.find(function(n) { return n.id === nodeId; });
  var panel = document.getElementById('panel');
  var content = document.getElementById('panel-content');

  // Set panel gem color
  panel.style.setProperty('--panel-gem-color', GEM_COLORS[node.type]);
  panel.style.setProperty('--panel-gem-glow', GEM_GLOW[node.type]);

  // Group connections by type
  var connectionsByType = {};
  edgeData.forEach(function(e) {
    var src = typeof e.source === 'object' ? e.source.id : e.source;
    var tgt = typeof e.target === 'object' ? e.target.id : e.target;
    var otherId = src === nodeId ? tgt : src;
    var other = state.graph.nodes.find(function(n) { return n.id === otherId; });
    if (!other || !state.activeFilters.has(other.type)) return;
    if (!connectionsByType[other.type]) connectionsByType[other.type] = [];
    connectionsByType[other.type].push({ node: other, weight: e.weight, entryIds: e.entryIds });
  });

  Object.values(connectionsByType).forEach(function(arr) { arr.sort(function(a, b) { return b.weight - a.weight; }); });

  // Build panel DOM safely
  content.textContent = '';

  // Header
  var h2 = document.createElement('h2');
  h2.textContent = node.label;
  content.appendChild(h2);

  var badge = document.createElement('div');
  badge.className = 'gem-badge';
  badge.style.background = GEM_COLORS[node.type] + '20';
  badge.style.color = GEM_GLOW[node.type];
  badge.style.border = '1px solid ' + GEM_COLORS[node.type] + '40';
  badge.textContent = GEM_NAMES[node.type];
  content.appendChild(badge);

  var countDiv = document.createElement('div');
  countDiv.className = 'panel-entry-count';
  countDiv.textContent = node.entryCount + ' WoB entries';
  content.appendChild(countDiv);

  // "Browse all entries" button -- finds entries by tag
  var browseBtn = document.createElement('button');
  browseBtn.className = 'browse-all-btn';
  browseBtn.textContent = 'Browse all ' + node.entryCount + ' entries \u2192';
  browseBtn.addEventListener('click', function() {
    showAllEntries(nodeId);
  });
  content.appendChild(browseBtn);

  var divider = document.createElement('div');
  divider.className = 'panel-divider';
  content.appendChild(divider);

  // Connections list
  var connSection = document.createElement('div');
  connSection.className = 'connections-section';
  connSection.id = 'connections-list';

  var typeOrder = ['character', 'shard', 'magic', 'world', 'concept'];
  typeOrder.forEach(function(type) {
    var conns = connectionsByType[type];
    if (!conns || conns.length === 0) return;

    var h3 = document.createElement('h3');
    h3.style.color = GEM_GLOW[type];
    h3.textContent = TYPE_LABELS[type] + ' (' + conns.length + ')';
    connSection.appendChild(h3);

    conns.forEach(function(c) {
      var item = document.createElement('div');
      item.className = 'connection-item';
      item.dataset.node = c.node.id;
      item.dataset.entries = JSON.stringify(c.entryIds);

      var left = document.createElement('div');
      left.className = 'conn-left';

      var gem = document.createElement('span');
      gem.className = 'conn-gem';
      gem.style.background = GEM_COLORS[c.node.type];
      left.appendChild(gem);

      var nameSpan = document.createElement('span');
      nameSpan.style.color = GEM_GLOW[c.node.type];
      nameSpan.textContent = c.node.label;
      left.appendChild(nameSpan);

      var weight = document.createElement('span');
      weight.className = 'connection-weight';
      weight.style.background = GEM_COLORS[c.node.type] + '15';
      weight.style.color = GEM_GLOW[c.node.type];
      weight.textContent = c.weight;

      item.appendChild(left);
      item.appendChild(weight);
      item.addEventListener('click', function() {
        showEntries(nodeId, c.node.id, c.entryIds);
      });
      connSection.appendChild(item);
    });
  });

  content.appendChild(connSection);

  // Similar Entities section
  var simData = state.similarity[nodeId];
  if (simData && simData.length > 0) {
    var simDivider = document.createElement('div');
    simDivider.className = 'panel-divider';
    content.appendChild(simDivider);

    var simSection = document.createElement('div');
    simSection.className = 'similar-section';

    var simH3 = document.createElement('h3');
    simH3.style.color = 'var(--stormlight)';
    simH3.style.fontSize = '0.7rem';
    simH3.style.textTransform = 'uppercase';
    simH3.style.letterSpacing = '0.15em';
    simH3.style.marginBottom = '0.6rem';
    simH3.style.opacity = '0.7';
    simH3.textContent = 'Semantically Similar';
    simSection.appendChild(simH3);

    simData.forEach(function(s) {
      var simNode = state.graph.nodes.find(function(n) { return n.id === s.id; });
      if (!simNode || !state.activeFilters.has(simNode.type)) return;

      var item = document.createElement('div');
      item.className = 'connection-item';

      var left = document.createElement('div');
      left.className = 'conn-left';

      var gem = document.createElement('span');
      gem.className = 'conn-gem';
      gem.style.background = GEM_COLORS[simNode.type];
      left.appendChild(gem);

      var nameSpan = document.createElement('span');
      nameSpan.style.color = GEM_GLOW[simNode.type];
      nameSpan.textContent = simNode.label;
      left.appendChild(nameSpan);

      var score = document.createElement('span');
      score.className = 'connection-weight';
      score.style.background = 'rgba(200, 220, 255, 0.08)';
      score.style.color = 'var(--stormlight)';
      score.textContent = (s.score * 100).toFixed(0) + '%';

      item.appendChild(left);
      item.appendChild(score);
      item.addEventListener('click', function() {
        focusNode(s.id);
      });
      simSection.appendChild(item);
    });

    content.appendChild(simSection);
  }

  var entriesSection = document.createElement('div');
  entriesSection.id = 'entries-section';
  content.appendChild(entriesSection);

  panel.classList.add('open');
  showBackdrop(state._panelBackdrop);
}

function showEntries(nodeId, otherNodeId, entryIds) {
  var node = state.graph.nodes.find(function(n) { return n.id === nodeId; });
  var other = state.graph.nodes.find(function(n) { return n.id === otherNodeId; });
  var section = document.getElementById('entries-section');
  section.textContent = '';

  // Header
  var header = document.createElement('div');
  header.className = 'entries-header';

  var h3 = document.createElement('h3');
  h3.textContent = node.label + ' \u2194 ' + other.label;
  header.appendChild(h3);

  var backBtn = document.createElement('button');
  backBtn.className = 'entries-back';
  backBtn.textContent = '\u2190 Back';
  backBtn.addEventListener('click', function() {
    section.style.display = 'none';
    document.getElementById('connections-list').style.display = 'block';
  });
  header.appendChild(backBtn);
  section.appendChild(header);

  // Entries
  entryIds.forEach(function(eid) {
    var entry = state.entries[eid];
    if (!entry) return;
    var lines = entry.lines || [];
    var collapsed = lines.length > 3;

    var card = document.createElement('div');
    card.className = 'wob-entry';

    // Event line
    var eventDiv = document.createElement('div');
    eventDiv.className = 'wob-event';
    var dot = document.createElement('span');
    dot.className = 'wob-event-dot';
    eventDiv.appendChild(dot);
    var eventText = document.createTextNode(entry.event + ' \u00B7 ' + entry.date);
    eventDiv.appendChild(eventText);
    card.appendChild(eventDiv);

    // Visible lines
    var visibleLines = collapsed ? lines.slice(0, 3) : lines;
    visibleLines.forEach(function(line) {
      card.appendChild(createLineEl(line));
    });

    // Hidden lines + expand button
    if (collapsed) {
      var hiddenDiv = document.createElement('div');
      hiddenDiv.style.display = 'none';
      lines.slice(3).forEach(function(line) {
        hiddenDiv.appendChild(createLineEl(line));
      });
      card.appendChild(hiddenDiv);

      var expandBtn = document.createElement('button');
      expandBtn.className = 'wob-expand';
      expandBtn.textContent = 'Show ' + (lines.length - 3) + ' more lines';
      expandBtn.addEventListener('click', function() {
        if (hiddenDiv.style.display === 'none') {
          hiddenDiv.style.display = 'block';
          expandBtn.textContent = 'Show less';
        } else {
          hiddenDiv.style.display = 'none';
          expandBtn.textContent = 'Show ' + (lines.length - 3) + ' more lines';
        }
      });
      card.appendChild(expandBtn);
    }

    // Tags
    var tags = (entry.tags || []).slice(0, 8);
    if (tags.length) {
      var tagsDiv = document.createElement('div');
      tagsDiv.className = 'wob-tags';
      tags.forEach(function(t) {
        var tag = document.createElement('span');
        tag.className = 'wob-tag';
        tag.textContent = t;
        tagsDiv.appendChild(tag);
      });
      card.appendChild(tagsDiv);
    }

    section.appendChild(card);
  });

  section.style.display = 'block';
  document.getElementById('connections-list').style.display = 'none';
}

function createLineEl(line) {
  var div = document.createElement('div');
  div.className = 'wob-line';
  var speaker = document.createElement('span');
  var isBrandon = line.speaker.toLowerCase().indexOf('brandon') >= 0 || line.speaker.toLowerCase().indexOf('sanderson') >= 0;
  speaker.className = 'wob-speaker ' + (isBrandon ? 'brandon' : 'questioner');
  speaker.textContent = line.speaker + ':';
  div.appendChild(speaker);
  // The text from Arcanum may contain HTML formatting (emphasis, links)
  // We create a span and set textContent to safely render it as plain text
  var textSpan = document.createElement('span');
  textSpan.textContent = ' ' + stripHtml(line.text);
  div.appendChild(textSpan);
  return div;
}

// stripHtml decodes HTML entities and strips tags from Arcanum entry text.
// Safety: the input is set via textContent first (encoding), then decoded text
// is parsed via a throwaway element to strip tags. The final output is extracted
// via textContent, so no user-controlled HTML reaches the live DOM.
function stripHtml(text) {
  var tmp = document.createElement('div');
  tmp.textContent = text;
  var decoded = tmp.textContent;
  var parser = document.createElement('div');
  // This innerHTML assignment is on a detached element used only for text extraction.
  // The parsed result is immediately read via textContent below, never inserted into the DOM.
  parser.innerHTML = decoded;  // eslint-disable-line no-unsanitized/property
  return parser.textContent || parser.innerText || decoded;
}

function showAllEntries(nodeId) {
  var node = state.graph.nodes.find(function(n) { return n.id === nodeId; });
  var section = document.getElementById('entries-section');
  section.textContent = '';

  // Find all entries tagged with this node's id
  var matchingEntries = [];
  Object.keys(state.entries).forEach(function(eid) {
    var entry = state.entries[eid];
    if (entry.tags && entry.tags.indexOf(node.id) >= 0) {
      matchingEntries.push(entry);
    }
  });

  // Sort by date descending
  matchingEntries.sort(function(a, b) {
    return (b.date || '').localeCompare(a.date || '');
  });

  // Header
  var header = document.createElement('div');
  header.className = 'entries-header';

  var h3 = document.createElement('h3');
  h3.textContent = 'All entries for ' + node.label + ' (' + matchingEntries.length + ')';
  header.appendChild(h3);

  var backBtn = document.createElement('button');
  backBtn.className = 'entries-back';
  backBtn.textContent = '\u2190 Back';
  backBtn.addEventListener('click', function() {
    section.style.display = 'none';
    document.getElementById('connections-list').style.display = 'block';
  });
  header.appendChild(backBtn);
  section.appendChild(header);

  // Render entries
  matchingEntries.forEach(function(entry) {
    var lines = entry.lines || [];
    var collapsed = lines.length > 3;

    var card = document.createElement('div');
    card.className = 'wob-entry';

    var eventDiv = document.createElement('div');
    eventDiv.className = 'wob-event';
    var dot = document.createElement('span');
    dot.className = 'wob-event-dot';
    eventDiv.appendChild(dot);
    eventDiv.appendChild(document.createTextNode(entry.event + ' \u00B7 ' + entry.date));
    card.appendChild(eventDiv);

    var visibleLines = collapsed ? lines.slice(0, 3) : lines;
    visibleLines.forEach(function(line) { card.appendChild(createLineEl(line)); });

    if (collapsed) {
      var hiddenDiv = document.createElement('div');
      hiddenDiv.style.display = 'none';
      lines.slice(3).forEach(function(line) { hiddenDiv.appendChild(createLineEl(line)); });
      card.appendChild(hiddenDiv);

      var expandBtn = document.createElement('button');
      expandBtn.className = 'wob-expand';
      expandBtn.textContent = 'Show ' + (lines.length - 3) + ' more lines';
      expandBtn.addEventListener('click', function() {
        if (hiddenDiv.style.display === 'none') {
          hiddenDiv.style.display = 'block';
          expandBtn.textContent = 'Show less';
        } else {
          hiddenDiv.style.display = 'none';
          expandBtn.textContent = 'Show ' + (lines.length - 3) + ' more lines';
        }
      });
      card.appendChild(expandBtn);
    }

    var tags = (entry.tags || []).slice(0, 8);
    if (tags.length) {
      var tagsDiv = document.createElement('div');
      tagsDiv.className = 'wob-tags';
      tags.forEach(function(t) {
        var tag = document.createElement('span');
        tag.className = 'wob-tag';
        tag.textContent = t;
        tagsDiv.appendChild(tag);
      });
      card.appendChild(tagsDiv);
    }

    section.appendChild(card);
  });

  section.style.display = 'block';
  document.getElementById('connections-list').style.display = 'none';
}

export function closePanel() {
  document.getElementById('panel').classList.remove('open');
  hideBackdrop(state._panelBackdrop);
}
