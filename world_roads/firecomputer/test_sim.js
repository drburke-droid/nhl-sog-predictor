// Cross-check the page's simulator against Python: node test_sim.js
const fs = require('fs');
const { FireField } = require('./fire_sim.js');
const html = fs.readFileSync(__dirname + '/../media/firecomputer/fire_computer.html', 'utf8');
const DATA = JSON.parse(html.split('const DATA = ')[1].split(';\nconst S = DATA.stats;')[0]);
const C = DATA.act3, F = new FireField(C.net);
function run(msg, nonce, key) {
  const v = {};
  for (let i = 0; i < 64; i++) { v[C.msg[i]] = Number((msg >> BigInt(i)) & 1n); v[C.state[i]] = Number((nonce >> BigInt(i)) & 1n); }
  for (let i = 0; i < 128; i++) v[C.key[i]] = Number((key >> BigInt(i)) & 1n);
  C.round.forEach((f, i) => v[f] = i === 0 ? 1 : 0);
  let s = F.start(v);
  for (let b = 0; b < 31; b++) { for (let k = 1; k < F.beatSteps; k++) s = F.step(s); s = F.step(s, true); }
  for (let k = 0; k < F.maxDepth(C.outputs); k++) s = F.step(s);
  return C.outputs.reduce((acc, f, i) => acc | (BigInt(s[f]) << BigInt(i)), 0n);
}
let ok = true;
for (const [p, k, c] of DATA.stats.cipher.test_vectors) {
  const got = run(0n, BigInt('0x' + p), BigInt('0x' + k)).toString(16).padStart(16, '0');
  console.log(p, k, got, got === c ? 'ok' : 'WRONG'); ok = ok && got === c;
}
const sm = DATA.stats.cipher.sample, text = sm.text.padEnd(32, ' ');
sm.secret.forEach((c, i) => {
  const m = BigInt('0x' + Buffer.from(text.slice(8 * i, 8 * i + 8)).toString('hex'));
  const got = run(m, BigInt(sm.nonce_base + i), BigInt('0x' + sm.key)).toString(16).padStart(16, '0');
  console.log('sample block', i, got, got === c ? 'ok' : 'WRONG'); ok = ok && got === c;
});
console.log('beat steps', F.beatSteps, ok ? 'ALL OK' : 'FAILED');
process.exit(ok ? 0 : 1);
