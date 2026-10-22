// The keeper-by-keeper simulation of field.py, for the page (and for test_sim.js under node).
// kind: one letter per fire ('i' input, 'm' memory, 'n' NAND); src: what each fire watches.
function FireField(net) {
  const n = net.kind.length;
  this.n = n;
  this.kind = net.kind;
  this.init = net.init;
  this.a = new Int32Array(n);
  this.b = new Int32Array(n);
  this.depth = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const k = net.kind[i];
    if (k === 'n') {
      this.a[i] = net.src[i][0];
      this.b[i] = net.src[i][1];
      this.depth[i] = 1 + Math.max(this.depth[this.a[i]], this.depth[this.b[i]]);
    } else if (k === 'm') {
      this.a[i] = net.src[i][0];
    }
  }
  // keeper steps per drum beat: the longest chain from memory back to memory, plus the copy itself
  let d = 0;
  for (let i = 0; i < n; i++) if (net.kind[i] === 'm') d = Math.max(d, this.depth[this.a[i]]);
  this.beatSteps = d + 1;
}

// values: {fire: 0/1} for input and memory fires; NAND fires start dark
FireField.prototype.start = function (values) {
  const s = Uint8Array.from(this.init);
  for (const f in values) s[f] = values[f];
  return s;
};

// one keeper step (about 2 s): everyone looks at the fires as they are now, then sets their own
FireField.prototype.step = function (s, drum) {
  const t = new Uint8Array(s);
  const kind = this.kind, a = this.a, b = this.b;
  for (let i = 0; i < this.n; i++) {
    const k = kind.charCodeAt(i);
    if (k === 110) t[i] = 1 - (s[a[i]] & s[b[i]]);       // 'n'
    else if (k === 109 && drum) t[i] = s[a[i]];          // 'm'
  }
  return t;
};

FireField.prototype.maxDepth = function (fires) {
  let d = 0;
  for (const f of fires) d = Math.max(d, this.depth[f]);
  return d;
};

if (typeof module !== 'undefined') module.exports = { FireField };
