// js/guide.js -- Worldhopper's Guide modal setup

export function setupGuide() {
  var overlay = document.getElementById('guide-overlay');
  var btn = document.getElementById('guide-btn');
  var closeBtn = document.getElementById('guide-close');

  btn.addEventListener('click', function() {
    overlay.style.display = 'flex';
    // Force reflow so transition fires
    overlay.offsetHeight;
    overlay.classList.add('open');
  });

  function closeGuide() {
    overlay.classList.remove('open');
    setTimeout(function() {
      overlay.style.display = 'none';
    }, 400);
  }

  closeBtn.addEventListener('click', closeGuide);
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) closeGuide();
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && overlay.classList.contains('open')) closeGuide();
  });
}
