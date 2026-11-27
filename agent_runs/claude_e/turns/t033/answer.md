Two things about MAP measurement matter here:
- About half the stays (14.4k) have an arterial line, with readings in about 95% of hours and the gaps at random. The rest have none and are measured by cuff.
- Whether an arterial reading is taken does not depend on MAP level.

So the norepinephrine readout comes only from arterial-line stays, and missing arterial readings are not a selection problem.

Next I'll estimate the direct norepinephrine response on arterial MAP. Within continuing infusions in arterial-line stays, I'll regress MAP at t+k on the rate at t, with:
- fixed effects for the exact arterial reading at t,
- the previous rate interacted with the titration band as controls.

That isolates the random part of the rate change. One difference from insulin: a norepinephrine rate change carries into later rates almost one-for-one, because titration works from the previous rate. So the total effect mixes the effects of many later doses. I'll estimate that carry-over and deconvolve it, as I did for insulin.

```python
A['nor'] = Gtr['ne'].values
def lp_ne(outkey, k, fill0=False):
    rows=[]
    for h in range(1, 25-k):
        y = A[outkey][:, h+1+k]
        if fill0: y = np.nan_to_num(y, nan=0.0)
        rows.append(np.c_[np.full(nS,h), A['nor'][:,h+1], A['nor'][:,h], A['ma'][:,h+1], y])
    R = np.concatenate(rows); R = R[~np.isnan(R).any(1)]
    d = pd.DataFrame(R, columns=['h','x','xp','m0','y'])
    d = d[(d.xp>0.02)&(d.xp<0.99)]
    band = pd.cut(d.m0,[0,65,75,500],right=False,labels=False).astype(int)
    for b_ in range(3): d[f'xp_b{b_}'] = d.xp*(band==b_)
    d['m0r'] = d.m0.round()
    b,se,n = fe_reg(d,'y','x','m0r', extra=['xp_b0','xp_b1','xp_b2'])
    return b[0], se[0], n
outN=[]
for k in range(1,11):
    outN.append((k,)+tuple(np.round(lp_ne('ma',k),3))+tuple(np.round(lp_ne('nor',k,fill0=True),4)))
TN = pd.DataFrame(outN, columns=['lag','ma_b','ma_se','ma_n','nor_b','nor_se','nor_n']); print(TN.to_string())
dN = deconv(TN.ma_b.values, TN.nor_b.values); print('direct NE->MAP per 1 mcg/kg/min, lags 1..10:', dN.round(2), ' sum', dN.sum().round(2))
```