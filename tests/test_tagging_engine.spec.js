const { test, expect } = require('@playwright/test');

test('tagging engine tests all pass', async ({ page }) => {
  test.setTimeout(90000); // entries.json is ~6 MB, loading can be slow
  await page.goto('/tests/test_tagging_engine.html');

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
  await page.goto('/tests/test_tagging_engine.html');

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
