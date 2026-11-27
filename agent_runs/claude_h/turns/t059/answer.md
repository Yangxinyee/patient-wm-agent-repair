The edited model gives exactly 0.000 on all four edits in the real rollout, so the dose-invariant edit has no leak. On these fresh stays, the original model's spurious responses are:

| Edit | Original model shift | Basis |
|---|---|---|
| Insulin ×2 | +0.052 | 39 stays |
| Insulin ×½ | −0.152 | 39 stays |
| Norepinephrine ×2 | +0.008 | only 5 stays, 56 stay-hours |
| Norepinephrine ×½ | −0.023 | only 5 stays, 56 stay-hours |

So the edit removes the original's small non-zero answers at no NLL cost, which is the right outcome if the effects are absent.

One signal still makes me uneasy. For insulin in hours ≥ 12, the evaluation window, the finger-glucose readout estimate was +1.9 ± 0.7, about 2.7σ. That is the largest deviation from zero I have seen, in the window that matters.

Lab glucose measures the same underlying glucose with independent measurement noise, so it is a fair corroboration test.
- **Supports a real late effect:** lab shows a similar positive effect in hours ≥ 12. I would then consider installing a small positive insulin effect.
- **Points to noise:** lab shows about 0 there. A stacked finger-plus-lab estimate by window will show what the pooled evidence supports.

```python
t0=time.time()
hrI=RI['ti']-1; hrL=RIL['ti']-1
for nm,lo,hi in [('early',2,11),('late',12,23),('all',2,23)]:
    Lf,Lfs,rof,rofs=window_stats(RI,(hrI>=lo)&(hrI<=hi),AwI)
    Ll,Lls,rol,rols=window_stats(RIL,(hrL>=lo)&(hrL<=hi),AwI)
    print('%-5s FINGER readout %.2f (%.2f) L3-7 %s | LAB readout %.2f (%.2f) L3-7 %s'%(nm,rof,rofs,Lf[2:7].round(2),rol,rols,Ll[2:7].round(2)))
# stacked finger+lab: per stay sums over both outcome types (common coefficient), bootstrap jointly
def stacked(msk_f,msk_l,Aw_,nb=300):
    parts=[]
    for Rz,m in [(RI,msk_f),(RIL,msk_l)]:
        si_=Rz['si'][m]; ep=Rz['ep'][m]; Ly=Rz['Ly'][:,m]; Ry=Rz['Ry'][:,m]; parts.append((si_,ep,Ly,Ry))
    us=np.unique(np.concatenate([p[0] for p in parts])); ns=len(us)
    nL=np.zeros((12,ns)); dL=np.zeros((12,ns)); nR=np.zeros((12,ns)); dR=np.zeros((12,ns))
    for i,(si_,ep,Ly,Ry) in enumerate(parts):
        inv=np.searchsorted(us,si_)
        for k in range(12):
            ok=~np.isnan(Ly[k]); np.add.at(nL[k],inv[ok],(ep*np.nan_to_num(Ly[k]))[ok]); np.add.at(dL[k],inv[ok],(ep**2)[ok])
            if i==0:
                ok=~np.isnan(Ry[k]); np.add.at(nR[k],inv[ok],(ep*np.nan_to_num(Ry[k]))[ok]); np.add.at(dR[k],inv[ok],(ep**2)[ok])
    L=nL.sum(1)/dL.sum(1); R=nR.sum(1)/dR.sum(1); ro=(deconv(L,R)*Aw_).sum()
    rng=np.random.default_rng(3); W=rng.multinomial(ns,np.ones(ns)/ns,size=nb).T.astype(float)
    Lb=(nL@W)/(dL@W); Rb=(nR@W)/(dR@W); rob=[(deconv(Lb[:,b],Rb[:,b])*Aw_).sum() for b in range(nb)]
    return ro,np.std(rob),L
for nm,lo,hi in [('early',2,11),('late',12,23),('all',2,23)]:
    ro,ros,L=stacked((hrI>=lo)&(hrI<=hi),(hrL>=lo)&(hrL<=hi),AwI); print('STACKED %-5s readout %.2f (%.2f) L1-8 %s'%(nm,ro,ros,L[:8].round(2)))
print('time',round(time.time()-t0,1))
```