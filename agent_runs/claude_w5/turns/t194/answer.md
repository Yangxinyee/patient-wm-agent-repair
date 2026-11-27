Three findings:
- **Separation is easy.** A linear score separates drug from non-drug bundles perfectly on held-out stays, with a wide margin (±0.84 on a ±1 scale).
- **No unit is dead.** The smallest units contribute about 3% of the median unit, so repurposing has a cost.
- **The target barely decodes.** Even restricted to drug bundles, a linear readout of the per-slot prelude input gets only R² 0.28 (insulin) and 0.36 (norepinephrine).

A likely reason for the last point: the expected-dose term depends on the state averaged over slots, while each per-slot vector holds only one slot. Because every slot is written along the same direction, only the slot-average of the gate matters.

Next, two checks:
1. Refit using slot-averaged features (prelude input, prelude hidden units, E), drug bundles only.
2. Test how linearly the raw rates can be read from the action-encoder output u alone, since a gate built on u is exactly zero when no drug is given.

High R² for the rate from u, and a better target fit from slot-averaged prelude features, would support splitting the gate into a dose part and a centring part.

```python
def ridge_val(Xr,Yr,Fr,lams,icpt=True):
    Xb=torch.cat([Xr,torch.ones(len(Xr),1,dtype=torch.float64)],1) if icpt else Xr; n=Xb.shape[1]; res=[]
    for lam in lams:
        Rr=torch.linalg.solve(Xb[Fr].T@Xb[Fr]+lam*torch.eye(n,dtype=torch.float64),Xb[Fr].T@Yr[Fr]); Pv=Xb@Rr; v=~Fr
        res.append((lam,)+tuple(round(1-float(((Pv[v,j]-Yr[v,j])**2).mean()/Yr[v,j].var()),3) for j in range(Yr.shape[1])))
    return res
VK=[('',1,1),('_i2',2,1),('_ih',.5,1),('_n2',1,2),('_nh',1,.5)]
Xm={k:[] for k in ['n3','ph','E','u']}; Ym=[]; Yr_=[]; Fm_=[]
for d in D:
    if not isdrug(d): continue
    at=bundles_B[d['sid']][d['k']][1]
    for sfx,fi,fn_ in VK:
        for key in ['n3','ph','E']: Xm[key].append(d[key+sfx].double().mean(0))
        atv=at if sfx=='' else scale(at,*VN[sfx[1:]])
        Xm['u'].append(enc2(atv)[1].double())
        Ym.append([fi*d['ri']-d['cI'],fn_*d['rn']-d['cN']]); Yr_.append([fi*d['ri'],fn_*d['rn']]); Fm_.append(d['sid'] in FIT)
Ym=torch.tensor(Ym,dtype=torch.float64); Yr_=torch.tensor(Yr_,dtype=torch.float64); Fm_=torch.tensor(Fm_)
for key in ['n3','ph','E']:
    print(key,'slot-mean tau R2',ridge_val(torch.stack(Xm[key]),Ym,Fm_,[1e-3,1e-2,1e-1,1]))
Xu=torch.stack(Xm['u']); print('u -> raw rate R2',ridge_val(Xu,Yr_,Fm_,[1e-4,1e-3,1e-2]))
print('n3 -> raw rate R2',ridge_val(torch.stack(Xm['n3']),Yr_,Fm_,[1e-3,1e-2]))
```