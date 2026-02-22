const { test, expect } = require('@playwright/test');
const path = require('path');
const http = require('http');
const fs = require('fs');

// Simple static file server for serving test files (fetch() requires HTTP, not file://)
function createServer(rootDir) {
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.css': 'text/css'
  };

  const server = http.createServer((req, res) => {
    const filePath = path.join(rootDir, decodeURIComponent(req.url));
    const ext = path.extname(filePath);
    const contentType = mimeTypes[ext] || 'application/octet-stream';

    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('Not found: ' + req.url);
        return;
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(data);
    });
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      resolve({ server, port, url: 'http://127.0.0.1:' + port });
    });
  });
}

let serverInfo;

test.beforeAll(async () => {
  // Serve the entire sanderson-lore-visualization directory
  const projectRoot = path.resolve(__dirname, '..');
  serverInfo = await createServer(projectRoot);
});

test.afterAll(async () => {
  if (serverInfo && serverInfo.server) {
    serverInfo.server.close();
  }
});

test('tagging engine tests all pass', async ({ page }) => {
  test.setTimeout(90000); // entries.json is ~6 MB, loading can be slow
  await page.goto(serverInfo.url + '/tests/test_tagging_engine.html');

  // Wait for the summary to be populated (indicates all tests have run).
  // entries.json is large (~6 MB) so allow generous timeout.
  await page.waitForFunction(() => {
    const summary = document.getElementById('summary');
    return summary && summary.textContent && summary.textContent.length > 0;
  }, { timeout: 60000 });

  // Count failures (excluding skips, which are expected when scores.json is absent)
  const failures = await page.locator('.test-fail').count();
  const passes = await page.locator('.test-pass').count();
  const skips = await page.locator('.test-skip').count();

  // Log summary for CI visibility
  const summary = await page.locator('#summary').textContent();
  console.log('Test summary:', summary);

  // If there are failures, collect their names and messages for a useful error
  if (failures > 0) {
    const failDetails = await page.locator('.test-fail').allTextContents();
    console.log('Failed tests:', failDetails);
  }

  expect(failures).toBe(0);
  expect(passes).toBeGreaterThan(0);
  console.log(passes + ' passed, ' + skips + ' skipped, ' + failures + ' failed');
});

test('tagging engine functions are accessible in page', async ({ page }) => {
  await page.goto(serverInfo.url + '/tests/test_tagging_engine.html');

  // Wait for the page to load scripts
  await page.waitForFunction(() => typeof window.computeEffectiveThreshold === 'function', { timeout: 10000 });

  // Verify that the engine functions are defined globally
  const fnNames = [
    'resolveScore',
    'computeEffectiveThreshold',
    'filterBySpecificity',
    'applyMarginFilter',
    'applyMustBridgeFilter',
    'rebuildEdges',
    'computeImplicitTags'
  ];

  for (const fn of fnNames) {
    const isDefined = await page.evaluate((name) => typeof window[name] === 'function', fn);
    expect(isDefined).toBe(true);
  }
});
