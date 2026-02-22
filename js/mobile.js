// js/mobile.js -- Mobile backdrop helpers

import * as state from './state.js';
import { unfocus } from './panel.js';

export function isMobile() {
  return window.matchMedia('(max-width: 768px)').matches;
}

function createBackdrop(closeFn) {
  var backdrop = document.createElement('div');
  backdrop.className = 'panel-backdrop';
  backdrop.addEventListener('click', closeFn);
  document.body.appendChild(backdrop);
  return backdrop;
}

export function showBackdrop(backdrop) {
  if (backdrop && isMobile()) {
    backdrop.classList.add('visible');
  }
}

export function hideBackdrop(backdrop) {
  if (backdrop) {
    backdrop.classList.remove('visible');
  }
}

export function setupBackdrops() {
  state.set_panelBackdrop(createBackdrop(function() { unfocus(); }));
  state.set_reviewBackdrop(createBackdrop(function() {
    document.getElementById('review-panel').classList.remove('open');
    hideBackdrop(state._reviewBackdrop);
  }));
  // z-index: side panel backdrop behind panel (z20), review behind review (z25)
  state._panelBackdrop.style.zIndex = '19';
  state._reviewBackdrop.style.zIndex = '24';
}
