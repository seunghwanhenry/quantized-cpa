import sys, time, numpy as np, pandas as pd, warnings, pickle
warnings.filterwarnings('ignore')
import os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src')); sys.path.insert(0,os.path.dirname(__file__))
from qcpa import quantize, acf, harris_from_r, discretized_normal_ml, polychoric_lag, pd_toeplitz
from scipy.linalg import toeplitz
from simulation import simulate, LOOPS
def parzen(M):
    k=np.arange(M+1)/M; w=np.where(k<=0.5,1-6*k**2+6*k**3,2*(1-k)**3); return w
def s2_of(r,p):
    r=pd_toeplitz(r[:p+1]); phi=np.linalg.solve(toeplitz(r[:p]),r[1:]); return 1-phi@r[1:]
def aic_p(r,neff):
    best=(np.inf,1)
    for p in range(1,len(r)):
        try: s2=s2_of(r,p)
        except Exception: continue
        if s2>0:
            c=neff*np.log(s2)+2*p
            if c<best[0]: best=(c,p)
    return best[1]
d=5; PMAX=20; rng=np.random.default_rng(7); rows=[]; t0=time.time(); store=[]
true={'well tuned':0.739,'sluggish':0.458,'aggressive':0.495}
for name,(Kc,Ti) in LOOPS.items():
    for n in [750,1500,3000,7000]:
        for rep in range(8):
            y=simulate(n,Kc,Ti,d=d,seed=9000+rep+n)
            for rho in [1,2,3]:
                q=rho*y.std(); yq=quantize(y,q,rng.uniform(0,q))
                mu,s=discretized_normal_ml(yq,q)
                rp=np.array([1.0]+[polychoric_lag(yq,k,q,mu,s) for k in range(1,PMAX+1)])
                store.append((name,n,rho,rep,rp))
                out=dict(loop=name,n=n,rho=rho,rep=rep,true=true[name])
                out['p5']=harris_from_r(rp[:6],d); out['p10']=harris_from_r(rp[:11],d)
                out['taper20']=harris_from_r(rp*parzen(20),d)
                out['taper20_p10']=harris_from_r((rp*parzen(20))[:11],d)
                for f in [3,5]:
                    p=aic_p(rp,n/f); out[f'aic_n{f}']=harris_from_r(rp[:p+1],d); out[f'aic_n{f}_p']=p
                rows.append(out)
    print(name,round(time.time()-t0),flush=True)
pickle.dump(store,open('rp_store.pkl','wb'))
D=pd.DataFrame(rows); D.to_pickle('rules.pkl')
rules=['p5','p10','taper20','taper20_p10','aic_n3','aic_n5']
res=[]
for (n,rho),g in D.groupby(['n','rho']):
    r={'n':n,'rho':rho}
    for k in rules: e=g[k]-g['true']; r[k]=f"{e.mean():+.3f}/{np.sqrt(np.mean(e**2)):.3f}"
    res.append(r)
print(pd.DataFrame(res).to_string(index=False))
print('aic p medians', D.groupby('n')[['aic_n3_p','aic_n5_p']].median().to_dict())
