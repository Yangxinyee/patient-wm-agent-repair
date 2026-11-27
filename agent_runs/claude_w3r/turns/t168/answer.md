My prelude attention reproduces the model's attention weights exactly. Next I'll collect head 2's output (per slot, 48 dims) at every bundle of the 100 training stays. I'll fit read vectors c_i and c_n through the origin, with target = the drug's start rate at bundles that start it and 0 at all other bundles. Then I'll check the residual noise separately at drug and non-drug bundles. Residual sd ≲ 0.1 rate units at both would make head 2 a usable gated source. A large non-drug residual would mean I need another gate.

```python
O2s, tg_i, tg_n, isdrug, grp = [], [], [], [], []
for s in sorted(recs.stay_id.unique()):
    Bs, rs, EUs = stay_data(int(s))
    for k in range(1, len(Bs)):
        A_, O_ = prelude_att(Bs, rs, EUs, k)
        at = Bs[k][1]
        ri = [val(a['rate']) for a in at if a['code']==ci and a['kind']==0]; rn = [val(a['rate']) for a in at if a['code']==cn and a['kind']==0]
        O2s.append(O_[2].numpy()); tg_i.append(ri[0] if ri else 0.0); tg_n.append(rn[0] if rn else 0.0); isdrug.append(bool(ri or rn)); grp.append(int(s))
O2s = np.stack(O2s); tg_i, tg_n, isdrug, grp = map(np.array, (tg_i, tg_n, isdrug, grp)); print(O2s.shape, isdrug.mean().round(3))
Xo = O2s.mean(1)   # slot-mean head-2 output
print('slot spread of O2 (rel)', float(np.median(np.abs(O2s - O2s.mean(1, keepdims=True)).max((1,2)) / (np.abs(Xo).max(1)+1e-9))))
from sklearn.linear_model import Ridge
for nm, tg in [('ins', tg_i), ('ne', tg_n)]:
    p = cvp(Ridge(alpha=1e-3, fit_intercept=False), Xo, tg, groups=grp, cv=GroupKFold(5)); r_ = tg - p
    print(nm, 'resid sd: drug-with', r_[tg>0].std().round(4), 'drug-without', r_[isdrug & (tg==0)].std().round(4), 'nondrug', r_[~isdrug].std().round(4),
          'nondrug max|r|', np.abs(r_[~isdrug]).max().round(3))
```