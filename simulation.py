"""Monte Carlo study of Sections 5.1-5.3 (Table 3, Figs. 2-3).
Three PI loops, n = 30000, rho = 0.5..5, 16 replications, p = 20, d = 5, constrained AR fit."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from qcpa import quantize, acf, marginal_ml_open, polychoric_acf
from common import simulate, LOOPS, md_index

def main(n=30000, reps=16, P=20, d=5, rhos=(0.5, 1, 2, 3, 4, 5)):
    rows = []
    for loop, (Kc, Ti) in LOOPS.items():
        rng = np.random.default_rng({'well tuned': 1, 'sluggish': 2, 'aggressive': 3}[loop])
        for rep in range(reps):
            y = simulate(n, Kc, Ti, d=d, seed=100 + rep); ref = md_index(acf(y, P), P, d)
            for rho in rhos:
                q = rho * y.std(); yq = quantize(y, q, rng.uniform(0, q))
                yc = yq - yq.mean(); g = np.array([yc.var()] + [np.mean(yc[:-k] * yc[k:]) for k in range(1, P + 1)]); g0 = g[0] - q * q / 12
                mu, s = marginal_ml_open(yq, q)
                rows.append(dict(loop=loop, rep=rep, rho=rho, ref=ref, standard=md_index(acf(yq, P), P, d),
                                 sheppard=md_index(g / g0, P, d) if g0 > 0 else np.nan,
                                 polychoric=md_index(polychoric_acf(yq, q, P, mu, s), P, d)))
    D = pd.DataFrame(rows); D.to_csv('simulation_results.csv', index=False)
    for c in ['standard', 'sheppard', 'polychoric']: D[c] -= D['ref']
    print(D.groupby(['loop', 'rho'])[['standard', 'sheppard', 'polychoric']].agg(['mean', 'std']).round(3).to_string())

if __name__ == '__main__':
    main()
