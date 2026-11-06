import sys, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src')); sys.path.insert(0,os.path.dirname(__file__))
from qcpa import quantize, acf, harris_from_r, discretized_normal_ml, polychoric_lag
from simulation import simulate, LOOPS
d=5; PMAX=20; rng=np.random.default_rng(11); rows=[]; t0=time.time()
true={}
for name,(Kc,Ti) in LOOPS.items():
    y=simulate(200000,Kc,Ti,d=d,seed=999); true[name]=harris_from_r(acf(y,40),d)
for name,(Kc,Ti) in LOOPS.items():
    for n in [750,1500,3000,7000]:
        R=12 if n<=1500 else 8
        for rep in range(R):
            y=simulate(n,Kc,Ti,d=d,seed=5000+rep+n)
            for rho in [1,2,3]:
                q=rho*y.std(); yq=quantize(y,q,rng.uniform(0,q))
                rs=acf(yq,PMAX); mu,s=discretized_normal_ml(yq,q)
                rp=np.array([1.0]+[polychoric_lag(yq,k,q,mu,s) for k in range(1,PMAX+1)])
                for p in [5,10,20]:
                    def H(r):
                        try: return harris_from_r(r[:p+1],d)
                        except Exception: return np.nan
                    rows.append(dict(loop=name,n=n,rho=rho,rep=rep,p=p,true=true[name],std=H(rs),poly=H(rp)))
    print(name,'done',round(time.time()-t0),flush=True)
D=pd.DataFrame(rows); D.to_pickle('nlen.pkl')
D['se']=D['std']-D['true']; D['pe']=D['poly']-D['true']
g=D.groupby(['p','n','rho']).agg(std_bias=('se','mean'),poly_bias=('pe','mean'),poly_rmse=('pe',lambda x:np.sqrt(np.nanmean(x**2)))).round(3)
print(g.to_string()); print({k:round(v,3) for k,v in true.items()})
