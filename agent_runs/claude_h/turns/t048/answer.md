This run complicates the clean "both absent" picture, though both new signals are fragile:

| Outcome | Implied ×2 readout (se) | Hours < 12 | Hours ≥ 12 |
|---|---|---|---|
| norepinephrine → cuff MAP | −0.47 (0.18) | −0.46 | −0.32 |
| norepinephrine → arterial MAP (previous run) | −0.24 (0.14) | −0.32 | +0.02 |
| insulin → finger glucose | +0.84 (0.49) | +0.06 | +1.91 (≈0.5) |

For norepinephrine, the cuff signal is 2.6 sigma and agrees in sign with arterial. But the lag-1 coefficients are near zero or positive. That argues against the obvious confound, where clinicians titrate on an unrecorded true MAP. That confound would hit the shortest lag hardest.

For insulin, the earlier jitter-only analyses gave about 0. One difference this time is that the sample includes omissions, where insulin was due but not given (about 3% of rows), and the controls are richer. If omissions are not random, for example skipped when glucose is about to fall, they would bias the estimate upward. The dose edit only scales doses that were actually given, so the relevant variation is the jitter among given doses.

Next I test that directly:
1. Rerun insulin on rows with insulin > 0 only (pure jitter).
2. Check whether omissions predict the future glucose trend.
3. For norepinephrine, use a combined MAP outcome (mean of arterial and cuff where available) for more power, and check stability across two random halves of stays.

- **Omissions explain the insulin signal:** the jitter-only insulin readout is about 0 and omission rows show an abnormal glucose trend.
- **A real norepinephrine effect:** the combined MAP outcome stays clearly negative in both halves and both periods.

```python
t0=time.time()
# 1) insulin jitter-only rows
irows_pos=irows&(INS0>0)
_=lp_analysis(INS0, INS, irows_pos, ifeats, GF, AwI, label='INS(>0 only)->finger')
# 2) omissions: base>0 per sliding scale but no insulin
def base_of(g):
    b=np.zeros_like(g); b[g>=150]=1; b[g>=180]=2; b[g>=220]=3; b[g>=260]=4; b[g>=300]=6; return b
B=base_of(np.nan_to_num(Gpol))
om=(B>0)&(INS0==0)&~np.isnan(Gpol); given=(B>0)&(INS0>0)
print('omission rate among base>0:', om.sum()/(om.sum()+given.sum()))
# compare future finger glucose change after omission vs given, same glucose bin (5 mg/dL) -- use residual from HGB fit of G(t+k) on features
si2,ti2=np.nonzero(irows&(B>0)); X2=np.stack([ifeats[n][si2,ti2] for n in ifeats],1); f2=(stay_ids[si2]%2)
omi=om[si2,ti2].astype(float)
for k in (1,2,3,6):
    tt=ti2+k; ok=tt<=25; y=np.full(len(si2),np.nan); y[ok]=GF[si2[ok],tt[ok]]
    pred=np.full(len(y),np.nan); okk=~np.isnan(y)
    for f in (0,1):
        tr=okk&(f2!=f); te=okk&(f2==f)
        pred[te]=HGB(max_iter=200,min_samples_leaf=100).fit(X2[tr],y[tr]).predict(X2[te])
    res=y-pred; m=okk
    d=res[m&(omi==1)].mean()-res[m&(omi==0)].mean(); se=np.sqrt(res[m&(omi==1)].var()/max(1,(m&(omi==1)).sum()))
    print(' k',k,'omission resid diff %.2f (se %.2f), n_om %d'%(d,se,(m&(omi==1)).sum()))
# omission vs trend
tr_=GF-lag(GF,1)
print('omission corr with recent finger trend:', np.corrcoef(om[irows&(B>0)&~np.isnan(tr_)].astype(float), tr_[irows&(B>0)&~np.isnan(tr_)])[0,1].round(4))
# 3) NE combined MAP outcome
MAPc=np.where(np.isnan(GA),GC,np.where(np.isnan(GC),GA,(GA+GC)/2))
ep_c,Lc,Rc=lp_analysis(NE0, NE, cont, feats, MAPc, Aw, label='NE->combined')
# random halves
half=(stay_ids*2654435761 % 2**32)%2
for hv in (0,1):
    _=lp_analysis(NE0, NE, cont&(half[:,None]==hv), feats, MAPc, Aw, nboot=50, label='NE->combined half%d'%hv)
print('time',round(time.time()-t0,1))
```