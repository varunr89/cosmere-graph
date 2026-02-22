const { test, expect } = require('@playwright/test');
const path = require('path');
const http = require('http');
const fs = require('fs');

// Simple static file server (same pattern as other test files)
function createServer(rootDir) {
  var mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.css': 'text/css'
  };

  var server = http.createServer(function(req, res) {
    var filePath = path.join(rootDir, decodeURIComponent(req.url));
    var ext = path.extname(filePath);
    var contentType = mimeTypes[ext] || 'application/octet-stream';

    fs.readFile(filePath, function(err, data) {
      if (err) {
        res.writeHead(404);
        res.end('Not found: ' + req.url);
        return;
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(data);
    });
  });

  return new Promise(function(resolve) {
    server.listen(0, '127.0.0.1', function() {
      var port = server.address().port;
      resolve({ server: server, port: port, url: 'http://127.0.0.1:' + port });
    });
  });
}

var serverInfo;

test.beforeAll(async function() {
  var projectRoot = path.resolve(__dirname, '..');
  serverInfo = await createServer(projectRoot);
});

test.afterAll(async function() {
  if (serverInfo && serverInfo.server) {
    serverInfo.server.close();
  }
});

// Helper: wait for the page to finish loading (past the loading screen)
async function waitForAppReady(page) {
  await page.waitForFunction(function() {
    var gc = document.getElementById('graph-container');
    return gc && gc.style.opacity === '1';
  }, { timeout: 30000 });
}

// Helper: click Apply and wait for computation to finish.
// Clears the previous result first so we can detect when the new one arrives.
async function clickApplyAndWait(page) {
  await page.evaluate(function() {
    window._lastImplicitResult = null;
  });
  await page.locator('#apply-embeddings-btn').click();
  await page.waitForFunction(function() {
    return window._lastImplicitResult && window._lastImplicitResult.edges;
  }, { timeout: 90000 });
}

// Helper: extract the numeric tag count from the stats text
async function getTagCount(page) {
  var text = await page.locator('#embedding-stats').textContent();
  var match = text.match(/^(\d+)\s+tags/);
  return match ? parseInt(match[1], 10) : null;
}


// ── E2E Integration Test: Full Embedding Controls Workflow ───────────────────

test('full embedding controls workflow with real data', async function({ page }) {
  test.setTimeout(120000);

  // ── Step 1: Open index.html, wait for graph to load ──────────────────────
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);

  // ── Step 2: Verify baseline graph has no implicit edges ──────────────────
  var implicitBefore = await page.locator('line.implicit-edge').count();
  expect(implicitBefore).toBe(0);

  // ── Step 3: Click Apply with default settings ────────────────────────────
  await clickApplyAndWait(page);

  // ── Step 4: Verify implicit edges appear (dashed, amber) ─────────────────
  var implicitAfter = await page.locator('line.implicit-edge').count();
  expect(implicitAfter).toBeGreaterThan(0);

  // Verify dashed stroke-dasharray
  var dasharray = await page.locator('line.implicit-edge').first().getAttribute('stroke-dasharray');
  expect(dasharray).toBe('6,4');

  // Verify amber stroke color
  var stroke = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return el.getAttribute('stroke');
  });
  expect(stroke).toBeTruthy();

  // Verify stats show tag count > 0
  var defaultTagCount = await getTagCount(page);
  expect(defaultTagCount).toBeGreaterThan(0);

  // ── Step 5: Change calibration percentile to 40, click Apply ─────────────
  // Plan specified "threshold 0.90" but the UI uses calibration percentile instead
  // of a global threshold slider. Higher percentile = stricter per-entity threshold.
  // Expand the tuning panel first
  await page.locator('#tuning-toggle-btn').click();
  await expect(page.locator('#tuning-panel')).toBeVisible();

  // Set calibration percentile slider to 40
  await page.locator('#slider-calibration-percentile').fill('40');
  await expect(page.locator('#value-calibration-percentile')).toHaveText('40');

  // Click Apply and wait for recomputation
  await clickApplyAndWait(page);

  // ── Step 6: Verify tag count changed (different from default) ────────────
  var tagCountAt40 = await getTagCount(page);
  expect(tagCountAt40).toBeGreaterThan(0);
  expect(tagCountAt40).not.toBe(defaultTagCount);

  // ── Step 7: Change min specificity to 5.0, click Apply ───────────────────
  await page.locator('#slider-min-specificity').fill('5');
  await expect(page.locator('#value-min-specificity')).toHaveText('5.0');

  await clickApplyAndWait(page);

  // ── Step 8: Verify even fewer tags (hubs excluded at high specificity) ───
  var tagCountHighSpec = await getTagCount(page);
  expect(tagCountHighSpec).toBeLessThan(tagCountAt40);

  // ── Step 9: Toggle edge layer to "Implicit only" ─────────────────────────
  await page.locator('input[name="edge-layer"][value="implicit"]').click();

  // Verify only dashed edges visible (explicit hidden)
  var explicitHidden = await page.locator('g.links line').first().evaluate(function(el) {
    return el.getAttribute('visibility') === 'hidden';
  });
  expect(explicitHidden).toBe(true);

  var implicitVisible = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return el.getAttribute('visibility') !== 'hidden';
  });
  expect(implicitVisible).toBe(true);

  // ── Step 10: Open review panel ───────────────────────────────────────────
  await page.locator('#review-toggle-btn').click();
  var reviewPanelOpen = await page.locator('#review-panel').evaluate(function(el) {
    return el.classList.contains('open');
  });
  expect(reviewPanelOpen).toBe(true);

  // ── Step 11: Verify it lists the implicit tags ───────────────────────────
  var reviewRows = page.locator('#review-panel .review-row');
  var rowCount = await reviewRows.count();
  expect(rowCount).toBeGreaterThan(0);

  // ── Step 12: Click confirm on first tag ──────────────────────────────────
  var firstRow = reviewRows.first();

  // Row should not be confirmed initially
  var isConfirmedBefore = await firstRow.evaluate(function(el) {
    return el.classList.contains('confirmed');
  });
  expect(isConfirmedBefore).toBe(false);

  await firstRow.locator('.review-confirm').click();

  // ── Step 13: Verify status updates to confirmed ──────────────────────────
  var isConfirmedAfter = await firstRow.evaluate(function(el) {
    return el.classList.contains('confirmed');
  });
  expect(isConfirmedAfter).toBe(true);

  // Verify reviewState is updated
  var key = await firstRow.evaluate(function(el) { return el.dataset.key; });
  var stateValue = await page.evaluate(function(k) { return window.reviewState[k]; }, key);
  expect(stateValue).toBe('confirmed');

  // ── Step 14: Toggle edge layer back to "Both" ────────────────────────────
  await page.locator('input[name="edge-layer"][value="both"]').click();

  // Verify both explicit and implicit edges are visible
  var explicitNowVisible = await page.locator('g.links line').first().evaluate(function(el) {
    return el.getAttribute('visibility') !== 'hidden';
  });
  expect(explicitNowVisible).toBe(true);

  var implicitStillVisible = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return el.getAttribute('visibility') !== 'hidden';
  });
  expect(implicitStillVisible).toBe(true);
});
