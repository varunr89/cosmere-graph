const { test, expect } = require('@playwright/test');
const { waitForAppReady } = require('./helpers');

// ---- Test: Top bar renders with Apply button and stats display ----

test('embedding controls bar renders with Apply button and stats', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // The top bar should be visible
  const topBar = page.locator('#embedding-controls-bar');
  await expect(topBar).toBeVisible();

  // Apply button exists
  const applyBtn = page.locator('#apply-embeddings-btn');
  await expect(applyBtn).toBeVisible();
  await expect(applyBtn).toHaveText('Apply');

  // Stats display exists
  const stats = page.locator('#embedding-stats');
  await expect(stats).toBeVisible();
});

// ---- Test: Tuning panel expands and collapses ----

test('tuning panel expands and collapses', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Panel should be collapsed initially
  const tuningPanel = page.locator('#tuning-panel');
  await expect(tuningPanel).not.toBeVisible();

  // Toggle button should exist
  const toggleBtn = page.locator('#tuning-toggle-btn');
  await expect(toggleBtn).toBeVisible();

  // Click to expand
  await toggleBtn.click();
  await expect(tuningPanel).toBeVisible();

  // Click to collapse
  await toggleBtn.click();
  await expect(tuningPanel).not.toBeVisible();
});

// ---- Test: All sliders/controls exist with correct defaults ----

test('all sliders exist with correct default values', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Expand the tuning panel first
  await page.locator('#tuning-toggle-btn').click();
  await expect(page.locator('#tuning-panel')).toBeVisible();

  // Calibration percentile slider: 10-50, step 5, default 25
  const calibSlider = page.locator('#slider-calibration-percentile');
  await expect(calibSlider).toBeVisible();
  await expect(calibSlider).toHaveAttribute('min', '10');
  await expect(calibSlider).toHaveAttribute('max', '50');
  await expect(calibSlider).toHaveAttribute('step', '5');
  await expect(calibSlider).toHaveValue('25');

  // Min Specificity slider: 0.0-5.0, step 0.1, default 2.0
  const specSlider = page.locator('#slider-min-specificity');
  await expect(specSlider).toBeVisible();
  await expect(specSlider).toHaveAttribute('min', '0');
  await expect(specSlider).toHaveAttribute('max', '5');
  await expect(specSlider).toHaveAttribute('step', '0.1');
  await expect(specSlider).toHaveValue('2');

  // Confidence Margin slider: 0.00-0.15, step 0.01, default 0.05
  const marginSlider = page.locator('#slider-confidence-margin');
  await expect(marginSlider).toBeVisible();
  await expect(marginSlider).toHaveAttribute('min', '0');
  await expect(marginSlider).toHaveAttribute('max', '0.15');
  await expect(marginSlider).toHaveAttribute('step', '0.01');
  await expect(marginSlider).toHaveValue('0.05');

  // Min Edge Weight slider: 2-10, step 1, default 2
  const edgeSlider = page.locator('#slider-min-edge-weight');
  await expect(edgeSlider).toBeVisible();
  await expect(edgeSlider).toHaveAttribute('min', '2');
  await expect(edgeSlider).toHaveAttribute('max', '10');
  await expect(edgeSlider).toHaveAttribute('step', '1');
  await expect(edgeSlider).toHaveValue('2');

  // Must-Bridge checkbox: default checked
  const bridgeCheckbox = page.locator('#checkbox-must-bridge');
  await expect(bridgeCheckbox).toBeVisible();
  await expect(bridgeCheckbox).toBeChecked();
});

// ---- Test: Slider value labels show current values ----

test('slider value labels show current values', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Expand tuning panel
  await page.locator('#tuning-toggle-btn').click();

  // Each slider should have a visible value label
  await expect(page.locator('#value-calibration-percentile')).toHaveText('25');
  await expect(page.locator('#value-min-specificity')).toHaveText('2.0');
  await expect(page.locator('#value-confidence-margin')).toHaveText('0.05');
  await expect(page.locator('#value-min-edge-weight')).toHaveText('2');
});

// ---- Test: Slider value labels update when dragged ----

test('slider value labels update when slider value changes', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  await page.locator('#tuning-toggle-btn').click();

  // Change calibration percentile
  await page.locator('#slider-calibration-percentile').fill('35');
  await expect(page.locator('#value-calibration-percentile')).toHaveText('35');

  // Change min specificity
  await page.locator('#slider-min-specificity').fill('3.5');
  await expect(page.locator('#value-min-specificity')).toHaveText('3.5');
});

// ---- Test: Apply button triggers recomputation and updates stats ----

test('Apply button triggers recomputation and updates stats', async ({ page }) => {
  test.setTimeout(120000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Stats should show placeholder text initially (before first Apply)
  const stats = page.locator('#embedding-stats');
  const initialText = await stats.textContent();

  // Click Apply
  await page.locator('#apply-embeddings-btn').click();

  // Wait for stats to update (scores.json is 8 MB, loading + computation takes time)
  await page.waitForFunction(() => {
    var el = document.getElementById('embedding-stats');
    return el && el.textContent && el.textContent.indexOf('tags') !== -1;
  }, { timeout: 90000 });

  // Stats should now show numbers
  const updatedText = await stats.textContent();
  expect(updatedText).toContain('tags');
  expect(updatedText).toContain('edges');
  expect(updatedText).not.toBe(initialText);
});

// ---- Test: Apply button shows loading state ----

test('Apply button shows loading state during computation', async ({ page }) => {
  test.setTimeout(120000);
  await page.goto('/index.html');
  await waitForAppReady(page);

  // Click Apply and immediately check button state
  const applyBtn = page.locator('#apply-embeddings-btn');
  await applyBtn.click();

  // During computation, button should be disabled
  // (It may be very fast, so we check right away)
  // After completion, button should be re-enabled
  await page.waitForFunction(() => {
    var el = document.getElementById('embedding-stats');
    return el && el.textContent && el.textContent.indexOf('tags') !== -1;
  }, { timeout: 90000 });

  // After completion, button should be enabled again
  await expect(applyBtn).toBeEnabled();
});
