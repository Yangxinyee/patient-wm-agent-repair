Bundle layout confirmed. Readings sit at integer hours. Drug bundles sit at h+0.0833 and hold the new START atoms (u) and the previous record's END atoms (e). prelude_attn returns attention of shape (4 heads × 4 slot queries × 7 keys), keys ordered [slots 0–3, e, u, z].

This run:
1. Measures where the slot queries attend on reading bundles versus drug bundles (20 stays).
2. Fits linear decoders on the normalised u token that output the insulin rate and the NE rate exactly, with zero on the empty-set token.

If attention to u is about 100% on both bundle types and the decoders are near-exact (R² ≈ 1, tiny errors), a value-projection edit gives an injection of exactly c × rate at drug bundles and nothing elsewhere. That is the clean, dose-linear building block for the edit.

```python
import time
t0 = time.time(); rng = np.random.default_rng(0)
selT = rng.choice(idsT, 1500, replace=False)
G3 = dict(tuple(tr[tr.stay_id.isin(selT)].groupby('stay_id')))
n1w, n1b = Wo['core.prelude.0.norm1.weight'], Wo['core.prelude.0.norm1.bias']
def lnp(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x-m)/np.sqrt(v+eps)*w + b
U, DI, DN, NA, EV = [], [], [], [], []
attR, attD = [], []
for j, s in enumerate(selT):
    df = G3[s]; B = bundles_of_stay(df, RU0)
    ins = df[df.event=='insulin']; nep = df[df.event=='norepinephrine']
    mI = dict(zip(np.round(ins.hour*60).astype(int), ins.rate)); mN = dict(zip(np.round(nep.hour*60).astype(int), nep.rate))
    for k, b in enumerate(B):
        key = int(round(b['t']*60)); U.append(b['u']); DI.append(mI.get(key,0.)); DN.append(mN.get(key,0.)); NA.append(b['nA'])
        if j < 200: EV.append(b['e'])
    if j < 20:
        St = run_stay(B)
        for k in range(1, len(B)):
            A, _ = prelude_attn(St[k-1], B, k); A = np.asarray(A).mean(1)  # heads x keys
            (attD if B[k]['nA']>0 else attR).append(A)
U = np.array(U); DI = np.array(DI); DN = np.array(DN); NA = np.array(NA); EV = np.array(EV)
for nm, L_ in [('reading', attR), ('drug', attD)]:
    A = np.mean(L_, 0); print(nm, 'n', len(L_), 'att per head [slots,e,u,z]:', [list(np.round([A[h,:4].sum(), A[h,4], A[h,5], A[h,6]],3)) for h in range(4)])
Xu = lnp(U, n1w, n1b); Xe = lnp(EV, n1w, n1b)
drug = NA>0; ue = U[~drug][0]; print('empty-u identical?', np.abs(U[~drug]-ue).max())
print('n drug tokens', drug.sum(), 'ins>0', (DI>0).sum(), 'ne>0', (DN>0).sum(), 'time', round(time.time()-t0,1))
```