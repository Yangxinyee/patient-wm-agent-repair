The start comparison cannot be trusted. My "placebo" cut-offs were contaminated, because their bandwidth spans the real 64/65 cut-off: the 61.5 cut-off showed jumps of +0.6 to +0.9. At the real cut-off, the arterial outcome shows nothing (+0.05 ± 0.21 at lag 1). Only cuff moves, by 1.6σ. So there is no reliable evidence of a start effect, and starts are irrelevant to ×2 and ×½ anyway.

Within ongoing titration there is no steep positive low-dose slope either:
- **Lag-1 slope by current rate:** −0.85 ± 0.77 at low rates, −0.01 ± 0.18 at medium, +0.10 ± 0.10 at high.
- **Implied ×2 readouts:** all mildly negative (−0.30, −0.48, −0.26).

So the picture is stable: a small, delayed, "worse-outcome" association for each drug. Insulin is +0.8 to +0.9 for ×2 (about 1.7σ) and norepinephrine about −0.34 (2.6σ). Both start around lag 3.

Before deciding, I want to check whether my bootstrap standard errors are honest. Future residualised outcomes are strongly correlated across adjacent hours and lags, so a mis-sized standard error would mislead me. A randomisation test needs no modelling assumptions. Because the true innovations are random noise, I can shuffle them across rows within strata (base level for insulin, previous-rate band for norepinephrine). I keep the residualised outcomes and the persistence profile fixed and recompute the implied readout. That gives the exact distribution of my statistic when there is no effect.

- **If the observed values sit in the tails (p < 0.05):** the association is real, whether causal or confounded.
- **If they sit inside the shuffled distribution:** the bootstrap overstated significance and "absent" is well supported.

```python
t0=time.time()
def perm_test(Rz, strata, Aw_, nperm=300, seed=0, label=''):
    ep=Rz['ep']; Ly=Rz['Ly']; Ry=Rz['Ry']
    okL=~np.isnan(Ly); Lyz=np.nan_to_num(Ly)
    def Lof(e):
        return np.array([(e[okL[k]]*Lyz[k][okL[k]]).sum()/(e[okL[k]]**2).sum() for k in range(12)])
    R=np.array([(ep[~np.isnan(Ry[j])]*Ry[j][~np.isnan(Ry[j])]).sum()/(ep[~np.isnan(Ry[j])]**2).sum() for j in range(12)])
    obs=(deconv(Lof(ep),R)*Aw_).sum()
    rng=np.random.default_rng(seed); groups=[np.nonzero(strata==g)[0] for g in np.unique(strata)]
    null=[]
    for p in range(nperm):
        e=ep.copy()
        for gi in groups: e[gi]=ep[rng.permutation(gi)]
        null.append((deconv(Lof(e),R)*Aw_).sum())
    null=np.array(null); pval=(np.abs(null)>=abs(obs)).mean()
    print('%s observed %.3f | null mean %.3f sd %.3f | two-sided p %.3f'%(label,obs,null.mean(),null.std(),pval))
    return null
bI_=base_of(Gpol)[RI['si'],RI['ti']]
nullI=perm_test(RI, bI_, AwI, label='INSULIN finger')
rb=np.digitize(NE0[RNf_R['si'],RNf_R['ti']-1],[0.2,0.4,0.6,0.8])
nullN=perm_test(RNf_R, rb, Aw, label='NE combined (rows w/ reading)')
print('time',round(time.time()-t0,1))
```