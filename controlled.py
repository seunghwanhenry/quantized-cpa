"""Controlled scenarios of Section 5.5 (Table 4, Fig. 5): offset, non-Gaussian innovations, oscillatory disturbance."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from qcpa import quantize, acf, marginal_ml_open, polychoric_acf, ar_order2
from common import simulate_scenario, md_index

SCEN = [('baseline', 'gauss', 'ar1', 'random'), ('offset: bin centre', 'gauss', 'ar1', 'centre'), ('offset: bin edge', 'gauss', 'ar1', 'edge'),
        ('t(4) innovations', 't4', 'ar1', 'random'), ('Gaussian mixture', 'mix', 'ar1', 'random'), ('oscillatory disturbance', 'gauss', 'ar2', 'random')]

def main(n=7000, reps=10, d=5):
    rng = np.random.default_rng(9); rows = []
    for lab, inn, dist, off in SCEN:
        for rep in range(reps):
            y = simulate_scenario(n, innov=inn, dist=dist, seed=500 + rep); y = y - y.mean()
            for rho in (1, 2, 3, 4):
                q = rho * y.std(); p = ar_order2(n, rho); K = max(p, 20)
                yq = quantize(y, q, {'random': rng.uniform(0, q), 'centre': 0.0, 'edge': q / 2}[off])
                mu, s = marginal_ml_open(yq, q)
                rows.append(dict(scenario=lab, rep=rep, rho=rho, ref=md_index(acf(y, K), p, d), standard=md_index(acf(yq, K), p, d),
                                 polychoric=md_index(polychoric_acf(yq, q, K, mu, s), p, d), switch=np.mean(np.diff(yq) != 0)))
    D = pd.DataFrame(rows); D.to_csv('controlled_results.csv', index=False); D['err'] = D.polychoric - D.ref
    print(D.groupby(['scenario', 'rho']).agg(rmse=('err', lambda x: np.sqrt(np.nanmean(x ** 2))), switch=('switch', 'median')).round(3).to_string())

if __name__ == '__main__':
    main()
