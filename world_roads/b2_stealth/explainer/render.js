// Usage:
//   node render.js stills <t1,t2,...> <outdir>
//   node render.js video <frameStart> <frameEnd> <out.mp4>
//   node render.js events <out.json>
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path'), { spawn } = require('child_process');

(async () => {
  const [mode, ...args] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('console', m => console.error('[page]', m.text()));
  page.on('pageerror', e => { console.error('[pageerror]', e.message); process.exitCode = 1; });
  await page.goto('file://' + path.join(__dirname, 'scene.html'));
  const tl = JSON.parse(fs.readFileSync(path.join(__dirname, 'timeline.json')));
  await page.evaluate(tl => window.init(tl), tl);
  const grab = t => page.evaluate(t => { window.render(t); return document.getElementById('c').toDataURL('image/png'); }, t);

  if (mode === 'stills') {
    const outdir = args[1]; fs.mkdirSync(outdir, { recursive: true });
    for (const t of args[0].split(',').map(Number)) {
      const d = await grab(t);
      fs.writeFileSync(path.join(outdir, `f_${t.toFixed(2).padStart(7, '0')}.png`), Buffer.from(d.split(',')[1], 'base64'));
    }
  } else if (mode === 'video') {
    const f0 = +args[0], f1 = +args[1], out = args[2];
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', '30', '-c:v', 'png', '-i', '-',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-tune', 'animation', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
    const t0 = Date.now();
    for (let f = f0; f < f1; f++) {
      const d = await grab(f / tl.fps);
      const buf = Buffer.from(d.split(',')[1], 'base64');
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if ((f - f0) % 300 === 0) console.error(`${out}: frame ${f - f0}/${f1 - f0} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    }
    ff.stdin.end(); await new Promise(r => ff.on('close', r));
  } else if (mode === 'events') {
    fs.writeFileSync(args[0], JSON.stringify(await page.evaluate(() => window.sfxEvents()), null, 1));
  }
  await browser.close();
})();
