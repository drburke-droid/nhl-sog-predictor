// Capture the neurogaze 3D face at a range of lid closures (0 = open, 1 = closed).
import pw from '/opt/node-tools/node_modules/playwright/index.js'; const { chromium } = pw;
import fs from 'node:fs';
const out = process.argv[2];
const N = 41;
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
await page.goto('http://127.0.0.1:8765/index.html#case=normal&side=B&view=eyes', { waitUntil: 'load' });
await page.waitForFunction(() => window.__gaze && document.querySelector('#loader') ? getComputedStyle(document.querySelector('#loader')).display === 'none' || document.querySelector('#loader').hidden : !!window.__gaze, null, { timeout: 120000 }).catch(() => {});
await page.waitForTimeout(4000);
await page.addStyleTag({ content: '#stage > *:not(canvas) { visibility: hidden !important; } #target { visibility: hidden !important; }' });
await page.waitForTimeout(300);
const stage = await page.$('#stage');
const box = await stage.boundingBox();
console.log('stage', JSON.stringify(box));
for (let i = 0; i < N; i++) {
  const c = i / (N - 1);
  await page.evaluate((c) => {
    const g = window.__gaze;
    g.sim.state.lid.R = c; g.sim.state.lid.L = c;
    g.setGaze(0, 0);
  }, c);
  await page.waitForTimeout(250);
  await page.screenshot({ path: `${out}/lid_${String(i).padStart(2, '0')}.png`, clip: box });
}
await browser.close();
