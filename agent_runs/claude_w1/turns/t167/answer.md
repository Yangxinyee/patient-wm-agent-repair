The harness works: 100 stays give 2,958 query rows in 5.8 s. The remaining error is only that lagX, prof and the injection directions were defined after the earlier failure point, so they never ran. This run defines them and repeats the measurement. The test is still the factual lag profile and cross-drug leakage of the own-direction head-2 injection, plus linear decoding of doubled doses.

```python
def lagX(drug, K=4):
    M = OT[drug]; return np.column_stack([np.where(cc-k>=0, M[ii, np.maximum(cc-k,0)], 0) for k in range(1,K+1)])
def prof(dmu):
    out = {}
    for cd in np.unique(codes):
        m = codes==cd; X = np.column_stack([LI[m], LN_[m]]); b, *_ = np.linalg.lstsq(X, dmu[m], rcond=None); out[cd[:12]] = np.round(b, 3)
    return out
LI, LN_ = lagX('ins'), lagX('ne')
PI_ = np.linalg.lstsq(Pj, rI, rcond=None)[0]; PN_ = np.linalg.lstsq(Pj, rN, rcond=None)[0]
Wo['core.prelude.0.attn.kv.weight'] = KV0.copy(); Wo['core.prelude.0.attn.kv.weight'][288:336] += np.outer(PI_, w_ins) + np.outer(PN_, w_ne)
_, mu1 = run_rows(H100); Wo['core.prelude.0.attn.kv.weight'] = KV0.copy()
print('own-dir x1 extra: lag coefs [ins l1..4 | ne l1..4] per code:')
for k, v in prof(mu1-mu0).items(): print(' ', k, v)
print('codes', dict(zip(*np.unique(codes, return_counts=True))))
df2 = G3[H100[0]].copy(); m = df2.event.isin(['insulin','norepinephrine']); df2.loc[m,'rate'] *= 2; df2.loc[m,'dose'] *= 2
Bx = bundles_of_stay(df2, RU0); Ux = np.array([b['u'] for b in Bx if b['nA']>0]); Xx = lnp(Ux, n1w, n1b)
print('x2 decode ins', np.round(Xx@w_ins,2)[:8], 'true', df2[df2.event=='insulin'].rate.values[:8]); print('x2 decode ne', np.round(Xx@w_ne,3)[:8], 'true', np.round(df2[df2.event=='norepinephrine'].rate.values[:8],3))
```