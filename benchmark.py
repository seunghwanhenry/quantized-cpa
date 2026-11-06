"""Re-quantization benchmark on public industrial loops (Section 6).
Data: StictionGPT plant data (Apache-2.0) and ISDB loops distributed with Damarla et al. (2023).
Run:  python scripts/benchmark.py   (downloads both repositories on first use)
"""
import os, sys, glob, io, zipfile, urllib.request, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qcpa import quantize, acf, harris_from_r, sheppard_index, polychoric_index
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
REPOS = {'StictionGPT': 'https://codeload.github.com/TerenceXue-tech/StictionGPT/zip/HEAD',
         'damarla': 'https://codeload.github.com/seshu-damarla/control-valve-stiction/zip/HEAD'}

def fetch():
    os.makedirs(DATA, exist_ok=True)
    for name, url in REPOS.items():
        if not os.path.isdir(os.path.join(DATA, name)):
            zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(os.path.join(DATA, name))

def load_loops():
    loops = []
    for kind in ['non-stiction_data', 'stiction_data']:
        for f in sorted(glob.glob(os.path.join(DATA, 'StictionGPT', '*', 'plant_data', kind, '*.txt'))):
            rows = [l.rstrip('\r\n').split('\t') for l in open(f, encoding='utf8', errors='ignore').readlines()[5:]]
            rows = [r for r in rows if len(r) >= 5]
            try:
                pv, sv, mode = (np.array([float(r[j]) for r in rows]) for j in (2, 3, 4))
            except ValueError:
                continue
            e = (pv - sv)[mode == 1] if (mode == 1).sum() > 300 else pv - sv
            loops.append(('stiction' if kind.startswith('stiction') else 'non-stiction', os.path.basename(f), e))
    x = pd.ExcelFile(glob.glob(os.path.join(DATA, 'damarla', '*', 'data.xlsx'))[0])
    for sh in x.sheet_names:
        d = x.parse(sh); loops.append(('ISDB', sh, (d.PV - d.SP).values.astype(float)))
    return loops

def main(ds=(1, 2, 3), p=10, reps=5, seed=42):
    fetch(); rng = np.random.default_rng(seed); out = []
    for group, tag, e in load_loops():
        e = e[np.isfinite(e)]; e = e - e.mean()
        if len(e) < 500 or e.std() == 0:
            continue
        steps = np.diff(np.unique(np.round(e, 6)))
        if len(steps) and np.median(steps) / e.std() > 0.3:
            continue                                   # already coarsely recorded: no reference
        for rho in (1, 2, 3):
            for r in range(reps):
                q = rho * e.std(); yq = quantize(e, q, rng.uniform(0, q))
                for d in ds:
                    out.append(dict(group=group, tag=tag, d=d, rho=rho, rep=r,
                                    ref=harris_from_r(acf(e, p), d),
                                    standard=harris_from_r(acf(yq, p), d),
                                    sheppard=sheppard_index(yq, q, d, p),
                                    polychoric=polychoric_index(yq, q, d, p)[0]))
    B = pd.DataFrame(out); B.to_csv('benchmark_results.csv', index=False)
    for c in ('standard', 'sheppard', 'polychoric'):
        B[c + '_abs_err'] = (B[c] - B['ref']).abs()
    print(B.groupby(['d', 'rho'])[[c + '_abs_err' for c in ('standard', 'sheppard', 'polychoric')]].median().round(3))

if __name__ == '__main__':
    main()
