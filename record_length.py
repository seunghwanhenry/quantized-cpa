"""Record-length study of Section 5.4 (Fig. 4): fixed p = 20 versus the order rule (Eq. 7)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from qcpa import quantize, acf, discretized_normal_ml, polychoric_lag, ar_order2
from common import simulate, LOOPS, md_index

def main(ns=(750, 1500, 3000, 7000), d=5, PMAX=20):
    true = {k: md_index(acf(simulate(200000, *v, d=d, seed=999), 40), 40, d) for k, v in LOOPS.items()}
    rng = np.random.default_rng(7); rng2 = np.random.default_rng(11); rows = []
    for loop, (Kc, Ti) in LOOPS.items():
        for n in ns:
            for rep in range(8):          # polychoric index: fixed p = 20 versus order rule (Fig. 4b,c)
                y = simulate(n, Kc, Ti, d=d, seed=9000 + rep + n)
                for rho in (1, 2, 3):
                    q = rho * y.std(); yq = quantize(y, q, rng.uniform(0, q)); mu, s = discretized_normal_ml(yq, q)
                    rp = np.array([1.0] + [polychoric_lag(yq, k, q, mu, s) for k in range(1, PMAX + 1)])
                    rows.append(dict(kind='polychoric', loop=loop, n=n, rho=rho, rep=rep, p20=md_index(rp, 20, d) - true[loop],
                                     rule=md_index(rp, ar_order2(n, rho), d) - true[loop]))
            for rep in range(12 if n <= 1500 else 8):   # standard index bias (Fig. 4a)
                y = simulate(n, Kc, Ti, d=d, seed=5000 + rep + n)
                for rho in (1, 2, 3):
                    q = rho * y.std(); yq = quantize(y, q, rng2.uniform(0, q))
                    rows.append(dict(kind='standard', loop=loop, n=n, rho=rho, rep=rep, standard=md_index(acf(yq, 20), 20, d) - true[loop]))
    D = pd.DataFrame(rows); D.to_csv('record_length_results.csv', index=False)
    rm = lambda x: np.sqrt(np.mean(x ** 2))
    print(D[D.kind == 'standard'].groupby(['n', 'rho']).standard.mean().round(3).to_string())
    print(D[D.kind == 'polychoric'].groupby(['n', 'rho']).agg(rmse_p20=('p20', rm), rmse_rule=('rule', rm)).round(3).to_string())

if __name__ == '__main__':
    main()
