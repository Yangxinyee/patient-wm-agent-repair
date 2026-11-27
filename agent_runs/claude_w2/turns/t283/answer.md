`design_build` is a deep edit:
- 4 gated units in the encoder's first MLP and 8 in its second;
- rewritten key/value and output weights in the prelude attention;
- a decoupled z_post.

It would still need parameter assembly, a check that the channel stays linear, a head readout, calibration and tests. That is roughly 15 or more of my 31 remaining runs, and run 265 showed the encoder's LayerNorm input scale varies a lot (cv 0.74), which threatens linearity.

`Dw` holds head inputs (`wcm`) for 300 stays under 8 insulin arms, collected under oracle-corrected rollouts. `rh` (same length) is probably the oracle's correction at each row. A cheaper route is to fit a linear readout from `wcm` to `rh` and put it in the value head. Test on held-out stays:
- **Viable:** R² is high, especially for the differences between arms at the same stay-hour.
- **Not viable:** low R² confirms the decision to build the channel.

```python
import numpy as np, pandas as pd, torch
def A(x): return x.double().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=float)
X = A(Dw['wcm']); cd = A(Dw['code']).astype(int); arm = A(Dw['arm']).astype(int); si = A(Dw['si']).astype(int)
hj = A(Dw['hj']).astype(int); t = A(Dw['t']); rh_ = A(rh)
print("codes:", {c: (int((cd==c).sum()), round(float(np.abs(rh_[cd==c]).mean()),3)) for c in np.unique(cd)})
print("rh mean by arm:", [round(float(rh_[arm==a].mean()),2) for a in range(8)])
gm = np.abs(rh_) > 0
gcodes = [c for c in np.unique(cd) if np.abs(rh_[cd==c]).mean() > 0.05]
m = np.isin(cd, gcodes)
tr, te = m & (si < 150), m & (si >= 150)
Xa = np.c_[X, np.ones(len(X))]
lam = 1e-2 * np.eye(Xa.shape[1]); lam[-1, -1] = 0
w = np.linalg.solve(Xa[tr].T @ Xa[tr] + lam * tr.sum(), Xa[tr].T @ rh_[tr])
pr = Xa @ w
def r2(y, p): return 1 - ((y-p)**2).mean() / y.var()
print("R2 train %.3f test %.3f" % (r2(rh_[tr], pr[tr]), r2(rh_[te], pr[te])))
df_ = pd.DataFrame(dict(si=si[te], hj=hj[te], arm=arm[te], y=rh_[te], p=pr[te]))
b0 = df_[df_.arm==0].set_index(['si','hj'])[['y','p']]
d1 = df_[df_.arm!=0].join(b0, on=['si','hj'], rsuffix='0').dropna()
dy, dp = d1.y-d1.y0, d1.p-d1.p0
print("arm-diff R2 %.3f  slope %.3f  n %d" % (r2(dy.values, dp.values), float(np.polyfit(dy, dp, 1)[0]), len(d1)))
print("mean dy/dp by arm:", d1.assign(dy=dy, dp=dp).groupby('arm')[['dy','dp']].mean().round(2).to_dict('index'))
```