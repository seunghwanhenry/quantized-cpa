"""End-to-end evaluation of Algorithm 1 on the industrial benchmark (Table 6): estimated rho, hybrid estimator,
multi-start fit, switching-rate and order-stability diagnostics, overall and by resolution ratio."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from qcpa import quantize, acf, marginal_ml_open, polychoric_acf, ar_order2, ar_fit_robust, harris_from_kappa
from common import load_loops

def main(d=2):
    loops = load_loops(); rows = []
    for group in ['ISDB', 'stiction', 'non-stiction']:
        rng = np.random.default_rng(43 if group == 'stiction' else 42)
        for g, tag, e in [L for L in loops if L[0] == group]:
            for rho in (1, 2, 3):
                for rep in range(5):
                    q = rho * e.std(); yq = quantize(e, q, rng.uniform(0, q)); mu, s = marginal_ml_open(yq, q)
                    nc = len(np.unique(np.round((yq - yq.min()) / q))); rh = q / s if nc >= 3 else np.nan
                    p = ar_order2(len(e), rh if np.isfinite(rh) else None); K = max(p, 20)
                    yc = yq - yq.mean(); gg = np.array([yc.var()] + [np.mean(yc[:-k] * yc[k:]) for k in range(1, K + 1)])
                    r_she = np.r_[1.0, gg[1:] / (gg[0] - q * q / 12)] if gg[0] > q * q / 12 else None
                    use_she = np.isfinite(rh) and rh <= 1 and r_she is not None and np.all(np.abs(r_she[1:]) < 1)
                    rc = r_she if use_she else polychoric_acf(yq, q, K, mu, s)
                    eta = harris_from_kappa(ar_fit_robust(rc, p, starts=6)[0], d)
                    eta_half = harris_from_kappa(ar_fit_robust(rc, max(2, p // 2), starts=3)[0], d)
                    ref = harris_from_kappa(ar_fit_robust(acf(e, K), p, starts=3)[0], d)
                    rows.append(dict(group=g, loop=tag, rho=rho, rep=rep, switch=np.mean(np.diff(yq) != 0), eta=eta, dord=abs(eta - eta_half), ref=ref))
    D = pd.DataFrame(rows); D['err'] = (D.eta - D.ref).abs(); D['reported'] = (D.switch >= 0.10) & (D.dord <= 0.10)
    D.to_csv('diagnostics_results.csv', index=False)
    def summ(v):
        r = v[v.reported]
        return pd.Series(dict(reported=v.reported.mean(), medae=r.err.median(), p95=r.err.quantile(.95), missed=(r.err > 0.1).mean(), unscreened_fail=(v.err > 0.1).mean()))
    print('all:', summ(D).round(3).to_dict())
    print(D.groupby('rho').apply(summ).round(3).to_string())

if __name__ == '__main__':
    main()
