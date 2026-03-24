# Stress Test Summary

| Test | Bets | Yield | Delta | MaxDD | Survives |
|------|------|-------|-------|-------|----------|
| BASELINE | 1158 | -1.4% | — | 41.5% | YES |
| Remove top 10% edges | 1043 | -2.8% | -1.4% | 45.9% | NO |
| Remove top 20% edges | 927 | -4.1% | -2.7% | 49.9% | NO |
| Add noise (std=0.02) | 1157 | -1.5% | -0.1% | 41.5% | NO |
| Add noise (std=0.05) | 1093 | -1.0% | +0.4% | 41.2% | NO |
| Vig increase +2% | 1158 | -2.4% | -1.0% | 44.9% | NO |
| Vig increase +5% | 1158 | -3.9% | -2.5% | 55.5% | NO |
