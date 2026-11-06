# Quantization-aware minimum-variance control performance assessment

Code accompanying *"Quantization-aware minimum-variance performance assessment of coarsely logged control loops"* (S. Song and J. Lee, submitted to Control Engineering Practice).

## Library
- `qcpa.py` — quantizer, open-ended discretized-normal marginal fit, lag-wise polychoric autocorrelation, constrained (partial-autocorrelation) AR fit with multi-start option, minimum-variance index, order rule (Eq. 7), Sheppard correction, and `corrected_index_v2` implementing Algorithm 1 (switching-rate and order-stability diagnostics).
- `common.py` — loop simulators (Sections 5.1 and 5.5) and loader for the public industrial data.

## Scripts reproducing the results
| Script | Paper | Approx. single-core time |
|---|---|---|
| `simulation.py` | Section 5.2–5.3, Table 3, Figs. 2–3 | ~10 min |
| `record_length.py` | Section 5.4, Fig. 4 | ~10 min |
| `controlled.py` | Section 5.5, Table 4, Fig. 5 | ~2 min |
| `benchmark.py` | Section 6.2, Table 5, Fig. 6 | ~3 min |
| `diagnostics.py` | Section 6.3, Table 6 (overall and by resolution ratio) | ~15 min |
| `bootstrap_coverage.py` | Section 4 (bootstrap coverage) | ~10 min |

`benchmark.py` and `diagnostics.py` download the public data into `data/` on first run; their results are reproduced exactly. The simulation scripts reproduce the reported values up to Monte Carlo error.

## Quick use
```python
from qcpa import corrected_index_v2
res = corrected_index_v2(y_logged, q=0.1, d=3)   # q: logging resolution, d: dead time in samples
res['eta']          # index, or None when flagged ('low code switching' / 'order-unstable fit')
res['pair']         # (corrected, standard) index pair, always returned
res['latent_acf']   # latent (polychoric) autocorrelation, always returned
```

## Data
- Plant loops: StictionGPT repository (T. Xue, C. Shang, D. Huang, B. Huang, Control Eng. Pract. 165 (2025) 106588), Apache-2.0.
- ISDB loops CHEM1, CHEM13, CHEM23: as distributed with S.K. Damarla et al., Ind. Eng. Chem. Res. 62 (2023) 4410–4421.
- The burn-in chamber logs of Section 7 are proprietary and not included.

## Reproduce
```bash
pip install -r requirements.txt
python benchmark.py
python diagnostics.py
python simulation.py
```

## License
MIT
