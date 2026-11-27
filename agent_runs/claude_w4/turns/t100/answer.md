The replica functions are all still there:
- `ins_atom(r)` and `ne_atom(r)` build atom embeddings;
- `u_set(atoms)` runs the drug set encoder.

Now I test whether u_k supports what the injection needs, using a grid of every kind of drug bundle (insulin only, NE only, both, none). Rates span 0 (the x0 edit keeps atoms at rate 0) up to 3x the recorded maximum.

1. Can a linear functional of u_k separate "insulin present" from "absent", and likewise for NE, with a clear margin?
2. Can a linear readout recover the insulin rate, and the NE rate, accurately even when the other drug is in the same bundle?

If both hold, gated linear units in z_post can carry the rate signal. If not, I will need piecewise units.

```python
import torch
maxI = tr.loc[tr.event=='insulin','rate'].max(); maxN = tr.loc[tr.event=='norepinephrine','rate'].max()
rI = np.unique(np.r_[0, 0.125, 0.25, np.arange(0.5, 15.01, 0.25)])
rN = np.unique(np.r_[0, 0.002, 0.005, np.arange(0.01, 1.0, 0.01), np.arange(1.0, 3.01, 0.05)])
rows_g = []
def U_of(atoms):
    v = u_set(atoms); return np.asarray(v.detach().numpy() if hasattr(v,'detach') else v, dtype=np.float64).ravel()
for a in rI: rows_g.append((a, np.nan, U_of([ins_atom(float(a))])))
for b in rN: rows_g.append((np.nan, b, U_of([ne_atom(float(b))])))
rng_ = np.random.default_rng(0)
for _ in range(3000):
    a = float(rng_.choice(rI)); b = float(rng_.choice(rN)); rows_g.append((a, b, U_of([ins_atom(a), ne_atom(b)])))
uE = U_of([]); rows_g.append((np.nan, np.nan, uE))
GI = np.array([r[0] for r in rows_g]); GN = np.array([r[1] for r in rows_g]); GU = np.stack([r[2] for r in rows_g])
print('maxI %.2f maxN %.3f grid %s  |u_empty - u_empty_old| %.2e' % (maxI, maxN, GU.shape, np.abs(uE - np.asarray(u_empty, dtype=np.float64)).max()))
```