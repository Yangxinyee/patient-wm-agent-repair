The grid run failed only because `u_set([])` cannot handle an empty atom list. For empty bundles I use the stored constant `u_empty` (the known no-action u_k) and rerun the same grid.

```python
def U_of(atoms):
    if len(atoms) == 0:
        v = u_empty
    else:
        v = u_set(atoms)
    return np.asarray(v.detach().numpy() if hasattr(v,'detach') else v, dtype=np.float64).ravel()
rows_g = []
for a in rI: rows_g.append((a, np.nan, U_of([ins_atom(float(a))])))
for b in rN: rows_g.append((np.nan, b, U_of([ne_atom(float(b))])))
rng_ = np.random.default_rng(0)
for _ in range(3000):
    a = float(rng_.choice(rI)); b = float(rng_.choice(rN)); rows_g.append((a, b, U_of([ins_atom(a), ne_atom(b)])))
rows_g.append((np.nan, np.nan, U_of([])))
GI = np.array([r[0] for r in rows_g]); GN = np.array([r[1] for r in rows_g]); GU = np.stack([r[2] for r in rows_g])
print('maxI %.2f maxN %.3f grid %s' % (maxI, maxN, GU.shape))
```