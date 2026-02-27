const { test, expect } = require('@playwright/test');
const { waitForAppReady, clickApplyAndWait, selectModelAndApply, getTagCount } = require('./helpers');

// -- Model Selection Tests ----------------------------------------------------

test('model selector populates from manifest', async function({ page }) {
  test.setTimeout(30000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  var modelSelect = page.locator('#model-select');
  await expect(modelSelect).toBeVisible();

  // Wait for manifest to load (options populated)
  await page.waitForFunction(function() {
    var sel = document.getElementById('model-select');
    return sel && sel.options.length > 0 && sel.options[0].text !== 'Loading models...';
  }, { timeout: 10000 });

  var optionCount = await modelSelect.locator('option').count();
  expect(optionCount).toBeGreaterThanOrEqual(2);
});

test('changing model and clicking Apply loads new scores', async function({ page }) {
  test.setTimeout(120000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Wait for manifest
  await page.waitForFunction(function() {
    var sel = document.getElementById('model-select');
    return sel && sel.options.length > 0 && sel.options[0].text !== 'Loading models...';
  }, { timeout: 10000 });

  // Apply with default model
  await clickApplyAndWait(page);
  var defaultTags = await getTagCount(page);
  expect(defaultTags).toBeGreaterThan(0);

  // Get all model options
  var options = await page.locator('#model-select option').evaluateAll(function(opts) {
    return opts.map(function(o) { return o.value; });
  });

  // If there's more than one model, switch to a different one
  if (options.length >= 2) {
    var currentModel = await page.locator('#model-select').inputValue();
    var otherModel = options.find(function(o) { return o !== currentModel; });

    await selectModelAndApply(page, otherModel);
    var otherTags = await getTagCount(page);
    expect(otherTags).toBeGreaterThan(0);

    // Stats text should include the model label
    var statsText = await page.locator('#embedding-stats').textContent();
    expect(statsText).toMatch(/^\[/); // starts with [ModelName]
  }
});

test('panel shows implicit connections after Apply', async function({ page }) {
  test.setTimeout(120000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Apply embeddings
  await clickApplyAndWait(page);

  // Find a node that has implicit edges
  var nodeId = await page.evaluate(function() {
    var result = window._lastImplicitResult;
    if (!result || !result.edges || result.edges.length === 0) return null;
    // Find an edge that is implicit or mixed
    for (var i = 0; i < result.edges.length; i++) {
      var e = result.edges[i];
      if (e.type === 'implicit' || e.type === 'mixed') {
        return e.source;
      }
    }
    return null;
  });

  if (nodeId) {
    // Click the node to open the panel
    await page.evaluate(function(id) {
      // Use the focusNode function from the app
      var event = new CustomEvent('focusNode', { detail: id });
      window.dispatchEvent(event);
    }, nodeId);

    // Focus the node via the app's exported function
    await page.evaluate(function(id) {
      // Find the node's SVG element and simulate a click
      var nodes = document.querySelectorAll('.gem-node');
      for (var i = 0; i < nodes.length; i++) {
        var d = nodes[i].__data__;
        if (d && d.id === id) {
          nodes[i].dispatchEvent(new MouseEvent('click', { bubbles: true }));
          break;
        }
      }
    }, nodeId);

    // Wait for panel to open
    await page.waitForSelector('#panel.open', { timeout: 5000 });

    // Check for implicit connections section
    var implicitSection = page.locator('#implicit-connections-list');
    var hasImplicit = await implicitSection.count();
    // It should exist (this node has implicit edges)
    expect(hasImplicit).toBe(1);

    // Check the section header mentions the model
    var headerText = await implicitSection.locator('.implicit-section-header').textContent();
    expect(headerText).toMatch(/^Discovered by /);
  }
});

test('implicit section hidden when embeddings not active', async function({ page }) {
  test.setTimeout(30000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Click any node without applying embeddings first
  await page.evaluate(function() {
    var nodes = document.querySelectorAll('.gem-node');
    if (nodes.length > 0) {
      nodes[0].dispatchEvent(new MouseEvent('click', { bubbles: true }));
    }
  });

  // Wait for panel to open
  await page.waitForSelector('#panel.open', { timeout: 5000 });

  // Implicit section should not exist
  var implicitSection = await page.locator('#implicit-connections-list').count();
  expect(implicitSection).toBe(0);
});
