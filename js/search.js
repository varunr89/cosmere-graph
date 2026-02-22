// js/search.js -- Search input, suggestions, and keyboard shortcuts

import { GEM_COLORS, GEM_GLOW, GEM_NAMES } from './constants.js';
import * as state from './state.js';
import { focusNode, unfocus } from './panel.js';

export function setupSearch() {
  var input = document.getElementById('search-input');
  var suggestions = document.getElementById('search-suggestions');
  var clearBtn = document.getElementById('search-clear');
  var activeIndex = -1;

  input.addEventListener('input', function() {
    var q = input.value.trim().toLowerCase();
    clearBtn.style.display = q ? 'block' : 'none';
    if (q.length < 1) { suggestions.style.display = 'none'; return; }

    var matches = state.graph.nodes
      .filter(function(n) { return n.id.indexOf(q) >= 0 || n.label.toLowerCase().indexOf(q) >= 0; })
      .sort(function(a, b) { return b.entryCount - a.entryCount; })
      .slice(0, 15);

    if (matches.length === 0) { suggestions.style.display = 'none'; return; }

    activeIndex = -1;
    suggestions.textContent = '';

    matches.forEach(function(m, i) {
      var div = document.createElement('div');
      div.className = 'suggestion';
      div.dataset.id = m.id;
      div.dataset.index = i;

      var nameSpan = document.createElement('span');
      nameSpan.style.color = GEM_GLOW[m.type];
      nameSpan.textContent = m.label;
      div.appendChild(nameSpan);

      var tagSpan = document.createElement('span');
      tagSpan.className = 'tag-type';
      tagSpan.style.background = GEM_COLORS[m.type] + '20';
      tagSpan.style.color = GEM_GLOW[m.type];
      tagSpan.textContent = GEM_NAMES[m.type];
      div.appendChild(tagSpan);

      div.addEventListener('click', function() {
        input.value = '';
        clearBtn.style.display = 'none';
        suggestions.style.display = 'none';
        focusNode(m.id);
      });
      suggestions.appendChild(div);
    });

    suggestions.style.display = 'block';
  });

  input.addEventListener('keydown', function(e) {
    var items = suggestions.querySelectorAll('.suggestion');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, items.length - 1);
      items.forEach(function(el, i) { el.classList.toggle('active', i === activeIndex); });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      items.forEach(function(el, i) { el.classList.toggle('active', i === activeIndex); });
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && items[activeIndex]) items[activeIndex].click();
      else if (items.length > 0) items[0].click();
    } else if (e.key === 'Escape') {
      suggestions.style.display = 'none';
      input.blur();
      unfocus();
    }
  });

  clearBtn.addEventListener('click', function() {
    input.value = '';
    clearBtn.style.display = 'none';
    suggestions.style.display = 'none';
    unfocus();
  });

  document.addEventListener('click', function(e) {
    if (!e.target.closest('#search-box')) suggestions.style.display = 'none';
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    if (e.key === '/' && !e.target.matches('input')) {
      e.preventDefault();
      document.getElementById('search-input').focus();
    }
    if (e.key === 'Escape') unfocus();
  });
}
