// tests/helpers.js

/**
 * Wait for the app to finish loading (loading screen fades, graph-container visible).
 * Used by: test_e2e, test_controls, test_hypothesis
 */
async function waitForAppReady(page) {
  await page.waitForFunction(() => {
    var gc = document.getElementById('graph-container');
    return gc && gc.style.opacity === '1';
  }, { timeout: 30000 });
}

/**
 * Click the Apply button and wait for implicit tag computation to finish.
 * Used by: test_e2e, test_hypothesis
 */
async function clickApplyAndWait(page) {
  await page.evaluate(function() {
    window._lastImplicitResult = null;
  });
  await page.locator('#apply-embeddings-btn').click();
  await page.waitForFunction(function() {
    return window._lastImplicitResult && window._lastImplicitResult.edges;
  }, { timeout: 90000 });
}

/**
 * Get the current implicit tag count from the stats display.
 * Used by: test_controls
 */
async function getTagCount(page) {
  return page.evaluate(function() {
    var text = document.getElementById('embedding-stats').textContent;
    var match = text.match(/(\d+)\s+tags/);
    return match ? parseInt(match[1], 10) : 0;
  });
}

/**
 * Select a model from the dropdown and click Apply, waiting for computation.
 * Used by: test_model_select
 */
async function selectModelAndApply(page, modelId) {
  await page.locator('#model-select').selectOption(modelId);
  await clickApplyAndWait(page);
}

module.exports = { waitForAppReady, clickApplyAndWait, getTagCount, selectModelAndApply };
