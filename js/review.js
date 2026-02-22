// js/review.js -- Review panel for implicit tags

import * as state from './state.js';
import { showBackdrop, hideBackdrop } from './mobile.js';

export function setupReviewPanel() {
  var toggleBtn = document.getElementById('review-toggle-btn');
  var panel = document.getElementById('review-panel');
  var closeBtn = document.getElementById('review-close-btn');
  var saveBtn = document.getElementById('save-reviews-btn');
  var loadBtn = document.getElementById('load-reviews-btn');
  var sortSelect = document.getElementById('review-sort-select');
  var filterCheckbox = document.getElementById('review-filter-unreviewed');

  toggleBtn.addEventListener('click', function() {
    var isOpen = panel.classList.contains('open');
    if (isOpen) {
      panel.classList.remove('open');
      hideBackdrop(state._reviewBackdrop);
    } else {
      panel.classList.add('open');
      showBackdrop(state._reviewBackdrop);
    }
  });

  closeBtn.addEventListener('click', function() {
    panel.classList.remove('open');
    hideBackdrop(state._reviewBackdrop);
  });

  saveBtn.addEventListener('click', function() {
    exportReviews();
  });

  loadBtn.addEventListener('click', function() {
    loadReviews();
  });

  sortSelect.addEventListener('change', function() {
    if (window._lastImplicitResult) {
      renderReviewRows(window._lastImplicitResult.implicitTags, sortSelect.value, filterCheckbox.checked);
    }
  });

  filterCheckbox.addEventListener('change', function() {
    if (window._lastImplicitResult) {
      renderReviewRows(window._lastImplicitResult.implicitTags, sortSelect.value, filterCheckbox.checked);
    }
  });

  // Event delegation for confirm/reject buttons (avoids 2*N listeners per render)
  var reviewTable = document.getElementById('review-table');
  reviewTable.addEventListener('click', function(e) {
    var btn = e.target;
    if (!btn.classList.contains('review-confirm') && !btn.classList.contains('review-reject')) return;
    var row = btn.closest('.review-row');
    if (!row) return;
    var key = row.dataset.key;
    if (btn.classList.contains('review-confirm')) {
      state.reviewState[key] = 'confirmed';
      row.classList.remove('rejected');
      row.classList.add('confirmed');
    } else {
      state.reviewState[key] = 'rejected';
      row.classList.remove('confirmed');
      row.classList.add('rejected');
    }
  });
}

export function populateReviewPanel(result) {
  var subtitle = document.getElementById('review-subtitle');
  subtitle.textContent = result.implicitTags.length + ' implicit tags from ' +
    result.stats.entitiesConsidered + ' entities';

  var sortSelect = document.getElementById('review-sort-select');
  var filterCheckbox = document.getElementById('review-filter-unreviewed');
  renderReviewRows(result.implicitTags, sortSelect.value, filterCheckbox.checked);
}

function renderReviewRows(implicitTags, sortMode, filterUnreviewed) {
  var table = document.getElementById('review-table');
  table.textContent = '';

  // Copy and optionally filter
  var sorted = implicitTags.slice();
  if (filterUnreviewed) {
    sorted = sorted.filter(function(tag) {
      var key = tag.entity + '::' + tag.entryId;
      return !state.reviewState[key];
    });
  }

  // Sort
  if (sortMode === 'score-desc') {
    sorted.sort(function(a, b) { return b.score - a.score; });
  } else if (sortMode === 'score-asc') {
    sorted.sort(function(a, b) { return a.score - b.score; });
  } else if (sortMode === 'entity-asc') {
    sorted.sort(function(a, b) { return a.entity.localeCompare(b.entity); });
  } else if (sortMode === 'status') {
    var statusOrder = { 'pending': 0, 'confirmed': 1, 'rejected': 2 };
    sorted.sort(function(a, b) {
      var sa = state.reviewState[a.entity + '::' + a.entryId] || 'pending';
      var sb = state.reviewState[b.entity + '::' + b.entryId] || 'pending';
      var diff = statusOrder[sa] - statusOrder[sb];
      if (diff !== 0) return diff;
      return b.score - a.score;
    });
  }

  for (var i = 0; i < sorted.length; i++) {
    var tag = sorted[i];
    var key = tag.entity + '::' + tag.entryId;
    var row = document.createElement('div');
    row.className = 'review-row';
    row.dataset.key = key;

    // Apply persisted review state
    if (state.reviewState[key] === 'confirmed') {
      row.classList.add('confirmed');
    } else if (state.reviewState[key] === 'rejected') {
      row.classList.add('rejected');
    }

    var entitySpan = document.createElement('span');
    entitySpan.className = 'review-entity';
    entitySpan.textContent = tag.entity;
    row.appendChild(entitySpan);

    // Show entry text (truncated) instead of entry ID
    var entrySpan = document.createElement('span');
    entrySpan.className = 'review-entry-id';
    var entryText = '';
    if (state.entries && state.entries[tag.entryId]) {
      var entry = state.entries[tag.entryId];
      if (entry.lines && entry.lines.length > 0) {
        entryText = entry.lines[0].text || '';
      } else if (entry.note) {
        entryText = entry.note;
      }
    }
    if (!entryText) {
      entryText = tag.entryId;
    }
    entrySpan.textContent = entryText.length > 40 ? entryText.substring(0, 40) + '...' : entryText;
    entrySpan.title = entryText;
    row.appendChild(entrySpan);

    var scoreSpan = document.createElement('span');
    scoreSpan.className = 'review-score';
    scoreSpan.textContent = tag.score.toFixed(3);
    row.appendChild(scoreSpan);

    var actionsCell = document.createElement('span');
    actionsCell.className = 'review-actions-cell';

    var confirmBtn = document.createElement('button');
    confirmBtn.className = 'review-confirm';
    confirmBtn.textContent = '\u2713';
    confirmBtn.title = 'Confirm';
    confirmBtn.setAttribute('aria-label', 'Confirm');
    actionsCell.appendChild(confirmBtn);

    var rejectBtn = document.createElement('button');
    rejectBtn.className = 'review-reject';
    rejectBtn.textContent = '\u2717';
    rejectBtn.title = 'Reject';
    rejectBtn.setAttribute('aria-label', 'Reject');
    actionsCell.appendChild(rejectBtn);

    row.appendChild(actionsCell);
    table.appendChild(row);
  }
}

function exportReviews() {
  var result = window._lastImplicitResult;
  if (!result) return;

  var exportData = {
    timestamp: new Date().toISOString(),
    totalTags: result.implicitTags.length,
    reviews: []
  };

  for (var i = 0; i < result.implicitTags.length; i++) {
    var tag = result.implicitTags[i];
    var key = tag.entity + '::' + tag.entryId;
    exportData.reviews.push({
      entity: tag.entity,
      entryId: tag.entryId,
      score: tag.score,
      status: state.reviewState[key] || 'pending'
    });
  }

  var blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'implicit-tag-reviews.json';
  a.click();
  URL.revokeObjectURL(url);
}

function loadReviews() {
  var input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.addEventListener('change', function(e) {
    var file = e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function(ev) {
      try {
        var data = JSON.parse(ev.target.result);
        if (data.reviews && Array.isArray(data.reviews)) {
          for (var i = 0; i < data.reviews.length; i++) {
            var r = data.reviews[i];
            if (r.entity && r.entryId && (r.status === 'confirmed' || r.status === 'rejected')) {
              state.reviewState[r.entity + '::' + r.entryId] = r.status;
            }
          }
          // Re-render if we have implicit results
          if (window._lastImplicitResult) {
            var sortSelect = document.getElementById('review-sort-select');
            var filterCheckbox = document.getElementById('review-filter-unreviewed');
            renderReviewRows(window._lastImplicitResult.implicitTags, sortSelect.value, filterCheckbox.checked);
          }
        }
      } catch (err) {
        console.error('Failed to load reviews:', err);
      }
    };
    reader.readAsText(file);
  });
  input.click();
}
