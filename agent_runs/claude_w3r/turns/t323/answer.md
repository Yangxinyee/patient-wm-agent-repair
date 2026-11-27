Orthogonalizing destroyed most of the rate signal (ins R²: 0.72 → 0.19; ne: 0.37 → 0.17). So in `out`, the rate direction is almost entirely confounded with glucose/MAP level. And NLL diff was +1.13 nats — way over budget. Not viable in this form.

I need to check whether a cleaner rate signal exists elsewhere in the state. The 992-dim state has: `out`, 4 `h` slots, `z` posterior. Rates in `u` were R²=0.99 at the prelude input, so after propagation they should be most intact in some slot of `h` rather than in `out` (which is a layer-normed average). I will:
- compare ridge R² of rate-vs-level predictors across `out`, each `h` slot, `mean(h)`, `z`;
- also regress the **delta** of state across action bundles on the bundle's rate, since that captures the causal response to the action.

Outcome that confirms: some `h` slot (or delta) gives much higher rate R² and lower gf R² than `out`.

Also drop the current (harmful) value-head edit for safety.

```python
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
for nm in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight']: drop_edit(nm)
print('after drop:', edited())
d=pickle.load(open('/work/rate_regs.pkl','rb')); OUT_=d['OUT_']; ok=d['ok']; RINS=d['RINS']; RNE=d['RNE']
tr=pd.read_csv('/data/train.csv', usecols=['stay_id','hour','event','value'])
S=set(np.unique(SST).tolist()); trs=tr[tr['stay_id'].isin(S)]
def lv(e):
    evs=trs[(trs['event']==e)&trs['value'].notna()][['stay_id','hour','value']].values; by={}
    for s,h,v in evs: by.setdefault(int(s),[]).append((h,v))
    for k in by: by[k].sort()
    r=np.full(len(SST),np.nan)
    for i in range(len(SST)):
        s=int(SST[i]); h=SHR[i]; l=by.get(s,[]); best=np.nan
        for h0,v in l:
            if h0<=h: best=v
            else: break
        r[i]=best
    return r
GF=lv('glucose_finger'); MAP=lv('map_arterial'); msk=ok & ~np.isnan(GF) & ~np.isnan(MAP); print('n=',int(msk.sum()))
H0,H1,H2,H3=SX[:,192:384],SX[:,384:576],SX[:,576:768],SX[:,768:960]; HM=(H0+H1+H2+H3)/4; Z=SX[:,960:]
print('Component R² (ridge α=100):')
for nm,X in (('out',OUT_),('h0',H0),('h1',H1),('h2',H2),('h3',H3),('hmean',HM),('z',Z),('out+hm',np.hstack([OUT_,HM])),('all',SX)):
    row=[Ridge(alpha=100).fit(X[msk],Y[msk]).score(X[msk],Y[msk]) for Y in (RINS,RNE,GF,MAP)]
    print(f'  {nm:8s}: ins={row[0]:.3f} ne={row[1]:.3f} gf={row[2]:.3f} map={row[3]:.3f}')
npz=np.load('/work/fwd_2/states_original.npz')
hi=npz['has_insulin']; hn=npz['has_norepinephrine']; sids=npz['stay_id']; hrs=npz['hour']
align=np.array_equal(sids,SST) and np.allclose(hrs,SHR); print('alignment ok:',align)
if align:
    order=np.lexsort((hrs,sids))
    Xs=SX[order]; sid_s=sids[order]; hi_s=hi[order]; hn_s=hn[order]; rins_s=RINS[order]; rne_s=RNE[order]
    same=(sid_s[:-1]==sid_s[1:]); D=Xs[1:]-Xs[:-1]; D=D[same]
    rins_c=rins_s[1:][same]; rne_c=rne_s[1:][same]; hi_c=hi_s[1:][same]; hn_c=hn_s[1:][same]
    HM_D=(D[:,192:384]+D[:,384:576]+D[:,576:768]+D[:,768:960])/4
    print('consec:',D.shape[0],'ins-act:',int(hi_c.sum()),'ne-act:',int(hn_c.sum()))
    for name,X_,m,Y in (('ΔOUT@ins',D[:,:192],hi_c,rins_c),('ΔHmean@ins',HM_D,hi_c,rins_c),
                        ('ΔOUT@ne',D[:,:192],hn_c,rne_c),('ΔHmean@ne',HM_D,hn_c,rne_c)):
        if m.sum()<30: continue
        r=Ridge(alpha=100).fit(X_[m],Y[m]); print(f'  {name}: R²={r.score(X_[m],Y[m]):.3f} |w|={np.linalg.norm(r.coef_):.2f}')
```