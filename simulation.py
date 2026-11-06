"""Monte Carlo study of Section 5 (three PI-controlled loops, rho = 0.5..5)."""
import os, sys, numpy as np, pandas as pd
from scipy import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qcpa import quantize, acf, harris_from_r, sheppard_index, polychoric_index

def simulate(n, Kc, Ti, d=5, a=0.8, b=0.2, phi=0.95, sig=0.1, seed=0):
    r = np.random.default_rng(seed); nz = signal.lfilter([1], [1, -phi], r.normal(0, sig, n + 500))
    y = np.zeros(n + 500); x = np.zeros(n + 500); u = np.zeros(n + 500); I = 0.0
    for t in range(1, n + 500):
        x[t] = a * x[t - 1] + b * (u[t - d] if t >= d else 0); y[t] = x[t] + nz[t]
        I += -y[t]; u[t] = Kc * (-y[t] + I / Ti)
    return y[500:]

LOOPS = {'well tuned': (0.5, 5.0), 'sluggish': (0.08, 25.0), 'aggressive': (1.0, 4.0)}

def main(n=30000, reps=8, p=20, d=5, rhos=(0.5, 1, 2, 3, 4, 5)):
    rng = np.random.default_rng(3); out = []
    for name, (Kc, Ti) in LOOPS.items():
        for rep in range(reps):
            y = simulate(n, Kc, Ti, d=d, seed=100 + rep); ref = harris_from_r(acf(y, p), d)
            for rho in rhos:
                q = rho * y.std(); yq = quantize(y, q, rng.uniform(0, q))
                out.append(dict(loop=name, rep=rep, rho=rho, ref=ref, standard=harris_from_r(acf(yq, p), d),
                                sheppard=sheppard_index(yq, q, d, p), polychoric=polychoric_index(yq, q, d, p)[0]))
    B = pd.DataFrame(out); B.to_csv('simulation_results.csv', index=False)
    print(B.groupby(['loop', 'rho']).mean(numeric_only=True).round(3))

if __name__ == '__main__':
    main()
