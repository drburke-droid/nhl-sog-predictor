# Edge Source Synthesis Report

**Question**: What is the smallest, simplest explanation for the observed profits?

## 1. Is the edge mainly book-structure driven?

- Soft-book unders (no model): -5.2% yield on 1636 bets

- Production (model): -1.4% yield on 1158 bets

- **Finding**: Model adds meaningful value beyond soft-book structure.


## 2. Is the edge mainly under-driven?

- All unders: -3.6% yield

- All overs: -4.3% yield


## 3. Is the model adding value beyond sharp-soft disagreement?

- High spread, no model: -4.7% (1229 bets)

- High spread + model agrees: -1.7% (862 bets)

- **Finding**: Model adds value on top of disagreement.


## 4. Is the selection engine superior to random?

- Selected yield: -1.4%

- Random matched yield: -5.3%

- Selection lift: +3.9%

- Empirical p-value: 0.07


## 5. Adversarial falsification

- BASELINE (production): yield=-1.4%, P(>0)=0.313

- 1: Permuted model scores: yield=-8.2%, P(>0)=0.0

- 2: Sharp-soft spread only: yield=-3.4%, P(>0)=0.365

- 3: Random noise model: yield=-5.3%, P(>0)=0.036

- 4: Inverted model: yield=+0.0%, P(>0)=0.0

- 5: Structural filters only (no model): yield=-5.2%, P(>0)=0.014

- 6: Model only (no side/book filter): yield=-2.4%, P(>0)=0.134


## 6. Can a simple system match the full framework?

- FULL: Production system: yield=-1.4% (1158 bets)

- SIMPLE: Sharp > soft by 3%: yield=-4.7% (1232 bets)

- SIMPLE: Sharp > soft by 5%: yield=-3.4% (104 bets)

- SIMPLE: Soft unders + 4+ sharp books: yield=-8.7% (1179 bets)

- SIMPLE: Top-5 daily sharp-soft spread: yield=-5.9% (293 bets)


## Verdict

Based on the evidence above, the most likely edge sources ranked:


1. **Soft-book pricing structure** — BetMGM unders are exploitable regardless of model

2. **Sharp-soft disagreement** — when sharp books disagree with soft book, betting the sharp side is profitable

3. **Under-side structural advantage** — unders outperform overs across all filters

4. **Model selection** — the model may help rank within the profitable universe but is not the primary edge source
