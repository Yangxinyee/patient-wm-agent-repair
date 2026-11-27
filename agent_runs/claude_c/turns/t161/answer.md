The states file has 4,356 bundles × 992 numbers (out, 4 h slots, z), with drug flags. Next I'll check whether h's direction is stable. I'll compute the cosine of each bundle's h (slot 0) with the mean h, and the same for the layer-normalised h, the form the prelude actually sees. If the cosines are near 1, a fixed direction can stand in for the slot tokens' content.

```python
X = S['x']; H = X[:, 192:384]; ins = S['has_insulin'].astype(bool); ne = S['has_norepinephrine'].astype(bool)
def stdz(A): A = A - A.mean(1, keepdims=True); return A / A.std(1, keepdims=True)
for nm, A in [('h', H), ('LN h', stdz(H))]:
    mu = A.mean(0); c = (A @ mu) / (np.linalg.norm(A,axis=1)*np.linalg.norm(mu))
    print(nm, 'cos to mean: pct 5/50/95', np.round(np.percentile(c,[5,50,95]),4), 'norm', round(np.linalg.norm(A,axis=1).mean(),1))
```