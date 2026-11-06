"""Shared simulators and data loader used by all scripts."""
import os, glob, io, zipfile, urllib.request, numpy as np, pandas as pd
from scipy import signal

LOOPS = {'well tuned': (0.5, 5.0), 'sluggish': (0.08, 25.0), 'aggressive': (1.0, 4.0)}

def simulate(n, Kc, Ti, d=5, a=0.8, b=0.2, phi=0.95, sig=0.1, seed=0):
    """First-order plant with dead time d, AR(1) disturbance, discrete PI control (Section 5.1)."""
    r = np.random.default_rng(seed); nz = signal.lfilter([1], [1, -phi], r.normal(0, sig, n + 500))
    y = np.zeros(n + 500); x = np.zeros(n + 500); u = np.zeros(n + 500); I = 0.0
    for t in range(1, n + 500):
        x[t] = a * x[t - 1] + b * (u[t - d] if t >= d else 0); y[t] = x[t] + nz[t]
        I += -y[t]; u[t] = Kc * (-y[t] + I / Ti)
    return y[500:]

def simulate_scenario(n, Kc=0.5, Ti=5.0, d=5, innov='gauss', dist='ar1', seed=0):
    """Controlled scenarios of Section 5.5: innovation law and disturbance dynamics."""
    r = np.random.default_rng(seed); N = n + 800
    if innov == 'gauss': e = r.normal(0, 1, N)
    elif innov == 't4': e = r.standard_t(4, N) / np.sqrt(2)
    else: e = np.where(r.uniform(size=N) < 0.1, r.normal(0, 3, N), r.normal(0, 1, N)) / np.sqrt(0.9 + 0.9)
    e *= 0.1
    nz = signal.lfilter([1], [1, -0.95], e) if dist == 'ar1' else signal.lfilter([1], [1, -2 * 0.95 * np.cos(0.15), 0.95 ** 2], e)
    y = np.zeros(N); x = np.zeros(N); u = np.zeros(N); I = 0.0
    for t in range(1, N):
        x[t] = 0.8 * x[t - 1] + 0.2 * (u[t - d] if t >= d else 0); y[t] = x[t] + nz[t]; I += -y[t]; u[t] = Kc * (-y[t] + I / Ti)
    return y[800:]

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REPOS = {'StictionGPT': 'https://codeload.github.com/TerenceXue-tech/StictionGPT/zip/HEAD',
         'damarla': 'https://codeload.github.com/seshu-damarla/control-valve-stiction/zip/HEAD'}

def fetch():
    os.makedirs(DATA, exist_ok=True)
    for name, url in REPOS.items():
        if not os.path.isdir(os.path.join(DATA, name)):
            zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(os.path.join(DATA, name))

def load_loops():
    """82 public industrial loops (Section 6.1): error in automatic mode, finely recorded loops only."""
    fetch(); loops = []
    for kind in ['non-stiction_data', 'stiction_data']:
        for f in sorted(glob.glob(os.path.join(DATA, 'StictionGPT', '*', 'plant_data', kind, '*.txt'))):
            rows = [l.rstrip('\r\n').split('\t') for l in open(f, encoding='utf8', errors='ignore').readlines()[5:]]
            rows = [r for r in rows if len(r) >= 5]
            try: pv, sv, mode = (np.array([float(r[j]) for r in rows]) for j in (2, 3, 4))
            except ValueError: continue
            e = (pv - sv)[mode == 1] if (mode == 1).sum() > 300 else pv - sv
            loops.append(('stiction' if kind.startswith('stiction') else 'non-stiction', os.path.basename(f), e))
    x = pd.ExcelFile(glob.glob(os.path.join(DATA, 'damarla', '*', 'data.xlsx'))[0])
    for sh in x.sheet_names:
        dd = x.parse(sh); loops.append(('ISDB', sh, (dd.PV - dd.SP).values.astype(float)))
    out = []
    for g, tag, e in loops:
        e = e[np.isfinite(e)]; e = e - e.mean()
        if len(e) < 500 or e.std() == 0: continue
        st = np.diff(np.unique(np.round(e, 6)))
        if len(st) and np.median(st) / e.std() > 0.3: continue
        out.append((g, tag, e))
    return out

def md_index(r, p, d):
    """Index of the stationary AR(p) closest to autocorrelation r (single start)."""
    from qcpa import ar_min_distance, harris_ar
    phi, _, v = ar_min_distance(r, p); return harris_ar(phi, v, d)
