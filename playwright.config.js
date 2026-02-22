// playwright.config.js
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 120_000,
  retries: 1,
  reporter: 'list',
  webServer: {
    command: 'python3 -m http.server 8080 --bind 127.0.0.1',
    port: 8080,
    reuseExistingServer: true,
  },
  use: {
    baseURL: 'http://127.0.0.1:8080',
  },
});
