Norepinephrine raises arterial MAP. A one-hour rate increase of 1 mcg/kg/min raises MAP by +3.0 mmHg (SE 0.11) at the next reading, and the effect builds to +5.7 by lag 3. The build-up is mostly because titration carries the higher rate forward (79% at the next hour, 63% two hours later).

Removing that carry-over gives a direct per-dose response of:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---|---|---|---|---|---|---|
| change (mmHg) | +3.05 | +2.51 | +1.81 | +0.94 | +0.49 | +0.23 | about 0 |

That sums to about +9.3 mmHg per sustained 1 mcg/kg/min. With typical rates of 0.5–1.0, doubling from hour 12 should raise MAP by several mmHg. The model currently says about 0.

The deconvolution here depends heavily on the carry-over, which is large. The rate cap at 1.0 also bends the relation between previous and current rate, so my linear controls may leave bias. Next I'll check robustness:
- control for the previous rate fully flexibly (fixed effects by band × previous rate);
- test linearity at low versus high rates, which matters because x2 pushes rates above anything observed;
- compute the implied true norepinephrine readouts on the tuning rows.

```python
def lp_ne2(outkey, k, fill0=False, sub=None):
    rows=[]
    for h in range(1, 25-k):
        y = A[outkey][:, h+1+k]
        if fill0: y = np.nan_to_num(y, nan=0.0)
        rows.append(np.c_[np.full(nS,h), A['nor'][:,h+1], A['nor'][:,h], A['ma'][:,h+1], y])
    R = np.concatenate(rows); R = R[~np.isnan(R).any(1)]
    d = pd.DataFrame(R, columns=['h','x','xp','m0','y']); d = d[(d.xp>0.02)&(d.xp<0.99)]
    if sub is not None: d = d[sub(d)]
    band = pd.cut(d.m0,[0,65,75,500],right=False,labels=False).astype(int)
    d['key'] = band*1000 + (d.xp*100).round()
    b,se,n = fe_reg(d,'y','x','key', extra=['m0']); return b[0], se[0], n
TG2 = np.array([lp_ne2('ma',k)[0] for k in range(1,11)]); TI2 = np.array([lp_ne2('nor',k,fill0=True)[0] for k in range(1,11)])
dN2 = deconv(TG2, TI2); print('flex FE: total', TG2.round(2), '\n  nor', TI2.round(3), '\n  direct', dN2.round(2), ' sum', dN2.sum().round(2))
lo = np.array([lp_ne2('ma',k, sub=lambda d: d.xp<0.4)[0] for k in [1,2]]); hi = np.array([lp_ne2('ma',k, sub=lambda d: d.xp>=0.6)[0] for k in [1,2]])
print('lag1-2 total, prev rate <0.4:', lo.round(2), '  prev rate >=0.6:', hi.round(2))
```