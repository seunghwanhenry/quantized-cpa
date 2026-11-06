"""Re-quantization benchmark of Section 6.2 (Table 5, Fig. 6). Downloads the public data on first run.
Loop-level errors: absolute errors averaged over the five quantizer offsets of each loop."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from qcpa import quantize, acf, marginal_ml_open, polychoric_acf, ar_min_distance, harris_ar, ar_order2
from common import load_loops

def idx_all(r, p):
    try:
        phi, _, v = ar_min_distance(r, p); return {d: harris_ar(phi, v, d) for d in (1, 2, 3)}
    except Exception:
        return {d: np.nan for d in (1, 2, 3)}

def main():
    loops = load_loops(); rows = []
    for group in ['ISDB', 'stiction', 'non-stiction']:
        rng = np.random.default_rng(42)
        for g, tag, e in [L for L in loops if L[0] == group]:
            for rho in (1, 2, 3):
                p = ar_order2(len(e), rho); K = max(p, 10)
                for rep in range(5):
                    q = rho * e.std(); yq = quantize(e, q, rng.uniform(0, q))
                    yc = yq - yq.mean(); gg = np.array([yc.var()] + [np.mean(yc[:-k] * yc[k:]) for k in range(1, K + 1)]); g0 = gg[0] - q * q / 12
                    mu, s = marginal_ml_open(yq, q)
                    A, S, H, P = idx_all(acf(e, K), p), idx_all(acf(yq, K), p), (idx_all(gg / g0, p) if g0 > 0 else {1: np.nan, 2: np.nan, 3: np.nan}), idx_all(polychoric_acf(yq, q, K, mu, s), p)
                    for d in (1, 2, 3):
                        rows.append(dict(group=g, loop=tag, rho=rho, rep=rep, d=d, ref=A[d], standard=S[d], sheppard=H[d], polychoric=P[d]))
    D = pd.DataFrame(rows); D.to_csv('benchmark_results.csv', index=False)
    B = D[D.d == 2].copy()
    for c in ['standard', 'sheppard', 'polychoric']: B[c] = (B[c] - B['ref']).abs()
    L = B.groupby(['group', 'loop', 'rho'])[['standard', 'sheppard', 'polychoric']].mean().reset_index()
    print(L.groupby('rho').agg(std_medae=('standard', 'median'), std_fail=('standard', lambda x: (x > 0.1).mean()),
                                poly_medae=('polychoric', 'median'), poly_p90=('polychoric', lambda x: x.quantile(.9)), poly_fail=('polychoric', lambda x: (x > 0.1).mean())).round(3).to_string())

if __name__ == '__main__':
    main()
