import { test, expect } from '@playwright/test';
import fs from 'fs';

test('stage 4', async ({ page }) => {
  const errors = [];
  page.on('console', m => { if (m.type()==='error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push('PAGEERROR: '+e.message));
  await page.goto('http://localhost:8000/');
  const txt = fs.readFileSync('../fixtures/body_measurements.yaml','utf8');
  await page.locator('textarea').fill(txt);
  await page.getByRole('button', { name: /Validate measurements/i }).click();
  await page.waitForTimeout(2000);
  await page.locator('button').filter({ hasText: 'Field definitions' }).click();
  await page.waitForTimeout(500);
  await page.getByRole('button', { name: /Resolve field definitions/i }).click();
  await page.waitForTimeout(2000);
  const confirmAll = page.getByRole('button', { name: /Confirm all offered definitions/i });
  if (await confirmAll.isEnabled()) await confirmAll.click();
  await page.waitForTimeout(1000);
  await page.locator('button').filter({ hasText: 'Base block & differences' }).click();
  await page.waitForTimeout(3000);
  console.log('ERRORS:', JSON.stringify(errors));
  await page.screenshot({ path: 'pwtest/stage4.png' });
});
