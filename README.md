# Quantization-aware minimum-variance control performance assessment

Code accompanying *"Quantization-aware minimum-variance performance assessment of coarsely logged control loops"* (S. Song and J. Lee, submitted to Control Engineering Practice).

## Contents
- `qcpa.py` — standard, Sheppard-corrected and polychoric minimum-variance indices; constrained autoregressive fit; reliability diagnostics (Algorithm 1); closed-form ceiling (Corollary 1).
- `simulation.py` — Monte Carlo study (Section 5, Table 3, Figs. 2–3).
- `record_length_nlen.py`, `record_length_rules.py` — record-length and order-rule study (Section 5.4, Fig. 4).
- `benchmark.py` — re-quantization benchmark on 82 public industrial loops (Section 6, Table 5, Fig. 6). Downloads the public data into `data/` on first run.

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
python simulation.py
python benchmark.py
```

## License
MIT
