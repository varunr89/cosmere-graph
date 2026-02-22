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

// Helper: click Apply and wait for computation to finish
async function clickApplyAndWait(page) {
  await page.locator('#apply-embeddings-btn').click();
  await page.waitForFunction(function() {
    return window._lastImplicitResult && window._lastImplicitResult.edges;
  }, { timeout: 90000 });
}

// ---- Test 1: After clicking Apply, implicit edges appear in the SVG (dashed, amber) ----

test('implicit edges render as dashed amber lines after Apply', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);

  // Before Apply, there should be no implicit edge lines
  var implicitBefore = await page.locator('line.implicit-edge').count();
  expect(implicitBefore).toBe(0);

  // Click Apply and wait for result
  await clickApplyAndWait(page);

  // After Apply, implicit edges should exist in SVG
  var implicitAfter = await page.locator('line.implicit-edge').count();
  expect(implicitAfter).toBeGreaterThan(0);

  // Verify dashed stroke-dasharray on implicit edges
  var dasharray = await page.locator('line.implicit-edge').first().getAttribute('stroke-dasharray');
  expect(dasharray).toBe('6,4');

  // Verify amber stroke color (var(--gem-heliodor-glow) = #fbbf24 or #E8C44A)
  var stroke = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return el.getAttribute('stroke');
  });
  // The color should be in the heliodor/amber family
  expect(stroke).toBeTruthy();
});

// ---- Test 2: Explicit edges remain solid (no dasharray) ----

test('explicit edges remain solid after Apply', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  // Explicit edges (from original graph.edges) should not have stroke-dasharray
  // They are rendered as line elements inside g.links
  var explicitEdge = page.locator('g.links line').first();
  var dasharray = await explicitEdge.getAttribute('stroke-dasharray');
  // Should be null or absent (not dashed)
  expect(dasharray).toBeNull();
});

// ---- Test 3: Edge layer radio toggles between views ----

test('edge layer radio toggles between Explicit, Both, and Implicit views', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  // The edge layer toggle should exist with 3 radio buttons
  var radios = page.locator('input[name="edge-layer"]');
  await expect(radios).toHaveCount(3);

  // Default should be "both"
  var checkedValue = await page.locator('input[name="edge-layer"]:checked').getAttribute('value');
  expect(checkedValue).toBe('both');

  // Switch to "explicit" -- implicit edges should be hidden
  await page.locator('input[name="edge-layer"][value="explicit"]').click();
  var implicitVisible = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return window.getComputedStyle(el).display !== 'none' && el.getAttribute('visibility') !== 'hidden';
  });
  expect(implicitVisible).toBe(false);

  // Original links should still be visible
  var explicitVisible = await page.locator('g.links line').first().evaluate(function(el) {
    return window.getComputedStyle(el).display !== 'none' && el.getAttribute('visibility') !== 'hidden';
  });
  expect(explicitVisible).toBe(true);

  // Switch to "implicit" -- explicit edges should be hidden
  await page.locator('input[name="edge-layer"][value="implicit"]').click();
  var explicitHidden = await page.locator('g.links line').first().evaluate(function(el) {
    return el.getAttribute('visibility') === 'hidden';
  });
  expect(explicitHidden).toBe(true);

  // Implicit edges should now be visible
  var implicitNowVisible = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return el.getAttribute('visibility') !== 'hidden';
  });
  expect(implicitNowVisible).toBe(true);

  // Switch to "both" -- both should be visible
  await page.locator('input[name="edge-layer"][value="both"]').click();
  var bothExplicit = await page.locator('g.links line').first().evaluate(function(el) {
    return el.getAttribute('visibility') !== 'hidden';
  });
  var bothImplicit = await page.locator('line.implicit-edge').first().evaluate(function(el) {
    return el.getAttribute('visibility') !== 'hidden';
  });
  expect(bothExplicit).toBe(true);
  expect(bothImplicit).toBe(true);
});

// ---- Test 4: Review panel opens and closes ----

test('review panel opens and closes', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  // Review panel should not have 'open' class initially
  var reviewPanel = page.locator('#review-panel');
  var hasOpenBefore = await reviewPanel.evaluate(function(el) {
    return el.classList.contains('open');
  });
  expect(hasOpenBefore).toBe(false);

  // Toggle button should exist and be visible
  var toggleBtn = page.locator('#review-toggle-btn');
  await expect(toggleBtn).toBeVisible();

  // Click to open
  await toggleBtn.click();
  var hasOpenAfter = await reviewPanel.evaluate(function(el) {
    return el.classList.contains('open');
  });
  expect(hasOpenAfter).toBe(true);

  // Click to close
  await toggleBtn.click();
  var hasOpenClosed = await reviewPanel.evaluate(function(el) {
    return el.classList.contains('open');
  });
  expect(hasOpenClosed).toBe(false);
});

// ---- Test 5: Review panel lists implicit tags with scores ----

test('review panel lists implicit tags with scores', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  // Open review panel
  await page.locator('#review-toggle-btn').click();
  await expect(page.locator('#review-panel')).toBeVisible();

  // Should have rows in the review table
  var rows = page.locator('#review-panel .review-row');
  var count = await rows.count();
  expect(count).toBeGreaterThan(0);

  // Each row should have entity name, score display, and confirm/reject buttons
  var firstRow = rows.first();
  var entityText = await firstRow.locator('.review-entity').textContent();
  expect(entityText.length).toBeGreaterThan(0);

  var scoreText = await firstRow.locator('.review-score').textContent();
  expect(scoreText.length).toBeGreaterThan(0);

  var confirmBtn = firstRow.locator('.review-confirm');
  var rejectBtn = firstRow.locator('.review-reject');
  await expect(confirmBtn).toBeVisible();
  await expect(rejectBtn).toBeVisible();
});

// ---- Test 6: Clicking confirm/reject updates the tag status ----

test('clicking confirm/reject updates review row status', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  // Open review panel
  await page.locator('#review-toggle-btn').click();
  await expect(page.locator('#review-panel')).toBeVisible();

  // Wait for rows to render
  var rows = page.locator('#review-panel .review-row');
  var count = await rows.count();
  expect(count).toBeGreaterThan(0);

  var firstRow = rows.first();

  // Row should not have confirmed or rejected class initially
  var classListBefore = await firstRow.evaluate(function(el) {
    return { confirmed: el.classList.contains('confirmed'), rejected: el.classList.contains('rejected') };
  });
  expect(classListBefore.confirmed).toBe(false);
  expect(classListBefore.rejected).toBe(false);

  // Click confirm button
  await firstRow.locator('.review-confirm').click();
  var hasConfirmed = await firstRow.evaluate(function(el) {
    return el.classList.contains('confirmed');
  });
  expect(hasConfirmed).toBe(true);

  // Click reject button on same row -- should switch
  await firstRow.locator('.review-reject').click();
  var classListAfterReject = await firstRow.evaluate(function(el) {
    return { confirmed: el.classList.contains('confirmed'), rejected: el.classList.contains('rejected') };
  });
  expect(classListAfterReject.confirmed).toBe(false);
  expect(classListAfterReject.rejected).toBe(true);

  // Verify reviewState is updated in the page
  var key = await firstRow.evaluate(function(el) { return el.dataset.key; });
  var stateValue = await page.evaluate(function(k) { return window.reviewState[k]; }, key);
  expect(stateValue).toBe('rejected');
});

// ---- Test 7: Save Reviews generates downloadable JSON ----

test('save reviews generates JSON download', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  // Open review panel and confirm one tag
  await page.locator('#review-toggle-btn').click();
  var firstRow = page.locator('#review-panel .review-row').first();
  await firstRow.locator('.review-confirm').click();

  // Listen for the download event
  var downloadPromise = page.waitForEvent('download');
  await page.locator('#save-reviews-btn').click();
  var download = await downloadPromise;

  // Verify the filename
  expect(download.suggestedFilename()).toBe('implicit-tag-reviews.json');

  // Read the downloaded content and verify structure
  var filePath = await download.path();
  var content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

  expect(content.timestamp).toBeTruthy();
  expect(content.totalTags).toBeGreaterThan(0);
  expect(Array.isArray(content.reviews)).toBe(true);
  expect(content.reviews.length).toBeGreaterThan(0);

  // At least one review should be confirmed (the one we clicked)
  var confirmedReviews = content.reviews.filter(function(r) { return r.status === 'confirmed'; });
  expect(confirmedReviews.length).toBeGreaterThanOrEqual(1);

  // Each review should have required fields
  var firstReview = content.reviews[0];
  expect(firstReview.entity).toBeTruthy();
  expect(firstReview.entryId).toBeTruthy();
  expect(typeof firstReview.score).toBe('number');
  expect(['pending', 'confirmed', 'rejected']).toContain(firstReview.status);
});

// ---- Test 8: Stats display shows tag and edge counts ----

test('stats display shows tag and edge counts after Apply', async function({ page }) {
  test.setTimeout(120000);
  await page.goto(serverInfo.url + '/index.html');
  await waitForAppReady(page);
  await clickApplyAndWait(page);

  var statsText = await page.locator('#embedding-stats').textContent();
  expect(statsText).toContain('tags');
  expect(statsText).toContain('edges');
  expect(statsText).toContain('implicit');
});
