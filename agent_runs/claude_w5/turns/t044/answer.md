`cv_r2`'s source is gone, but the decoding table `D` survives. It has 1,614 glucose prediction points from the 100 states stays, each with the state row and the insulin and norepinephrine doses at lags 1–3 plus the lag-1 residuals (dose minus the policy mean).

Next I'll redo the decoding myself: a cross-validated ridge regression, grouped by stay, of each dose target on the model state (`out`, the h slots, z). I'll also print the shapes of my old panel variables so I can extend to longer lags.
- High R² for residual doses (above about 0.8) would support a head-level readout edit.
- Low R² beyond lag 1 would push me toward building a dose memory inside the recurrent core.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
def cvr2(Xf, yv, groups, alphas=(1,10,100,1000)):
    best=None
    for a in alphas:
        pred=np.zeros_like(yv)
        for tri,tei in GroupKFold(5).split(Xf,yv,groups):
            mu=Xf[tri].mean(0); sd=Xf[tri].std(0)+1e-6
            mdl=Ridge(alpha=a).fit((Xf[tri]-mu)/sd, yv[tri])
            pred[tei]=mdl.predict((Xf[tei]-mu)/sd)
        r2=1-np.mean((yv-pred)**2)/np.var(yv)
        if best is None or r2>best[0]: best=(r2,a)
    return best
rows_=D['row'].values; grp=D['sid'].values
Xall=Z[rows_]
for tg in ['g_last','ins1','ins2','ins3','insres1','ne1','ne2','ne3','neres1']:
    yv=D[tg].values.astype(float)
    r_all=cvr2(Xall,yv,grp); r_out=cvr2(Xall[:,:192],yv,grp); r_h=cvr2(Xall[:,192:960],yv,grp)
    print(tg, 'all %.3f(a=%g)'%r_all, 'out %.3f'%r_out[0], 'h %.3f'%r_h[0])
for nm_ in ['F','L','R','Rres','Nres','Rn','sid','H','Gn','Mg','grid','P','PT','ins','ne','Z','st_sid','st_hr']:
    if nm_ in globals():
        v=globals()[nm_]; print(nm_, type(v).__name__, getattr(v,'shape',None) if not callable(v) else 'func')
```