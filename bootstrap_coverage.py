"""Coverage check of the moving-block bootstrap interval (Section 4): well-tuned loop, n = 3000, rho = 2, 22 records,
100 replicates, blocks of 300 samples, single-start fit in each replicate."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from qcpa import quantize, acf, marginal_ml_open, polychoric_acf, ar_order2
from common import simulate, md_index

def main(R=22, B=100, L=300, n=3000, rho=2.0, d=5):
    Kc, Ti = 0.5, 5.0; p = ar_order2(n, rho); K = max(p, 20)
    target = md_index(acf(simulate(200000, Kc, Ti, d=d, seed=4242), K)[:K + 1], p, d)
    rng = np.random.default_rng(1); hits = 0; sds = []; ests = []
    for rep in range(R):
        y = simulate(n, Kc, Ti, d=d, seed=100 + rep); q = rho * y.std(); yq = quantize(y, q, rng.uniform(0, q))
        mu, s = marginal_ml_open(yq, q); est = md_index(polychoric_acf(yq, q, K, mu, s), p, d); bs = []
        for b in range(B):
            st = rng.integers(0, n - L, n // L); yb = np.concatenate([yq[i:i + L] for i in st])
            mb, sb = marginal_ml_open(yb, q); bs.append(md_index(polychoric_acf(yb, q, K, mb, sb), p, d))
        lo, hi = np.percentile(bs, [2.5, 97.5]); hits += lo <= target <= hi; sds.append(np.std(bs)); ests.append(est)
    print(f'coverage {hits}/{R} = {hits / R:.2f}; bootstrap s.d. {np.mean(sds):.3f}; empirical s.d. {np.std(ests):.3f}; bias {np.mean(ests) - target:+.3f}')

if __name__ == '__main__':
    main()
