The reload worked: b1 and b2 were stale, and W1, W2 and TR were correct. The new error is only in my wrapper. head_mean returns a pair, the means plus the bin probabilities, not a single array. I fix run_rows to pick the 1-D means and re-run the same measurement. This run tests the same thing as before: the lag profile and cross-drug leakage of the own-direction head-2 injection, plus linear decoding of doubled doses.

```python
def run_rows(stays):
    rows = [r for s in stays for r in gen_samples(G3[s], pert=False)]
    res = head_mean(rows)
    if isinstance(res, (tuple, list)):
        mu = [np.asarray(x) for x in res if np.ndim(x)==1 and len(x)==len(rows)][0]
    else: mu = np.asarray(res)
    return rows, mu
t0 = time.time(); rows0, mu0 = run_rows(H100); print('rows', len(rows0), 'code ex', rows0[0][:3], 'mu ex', np.round(mu0[:3],1), 'secs %.1f' % (time.time()-t0))
codes = np.array([str(r[0]) for r in rows0]); ii = idxT[[r[1] for r in rows0]].values; cc = np.round([r[2] for r in rows0]).astype(int)+1
LI, LN_ = lagX('ins'), lagX('ne')
Wo['core.prelude.0.attn.kv.weight'] = KV0.copy(); Wo['core.prelude.0.attn.kv.weight'][288:336] += np.outer(PI_, w_ins) + np.outer(PN_, w_ne)
_, mu1 = run_rows(H100); Wo['core.prelude.0.attn.kv.weight'] = KV0.copy()
print('own-dir x1 extra: lag coefs [ins l1..4 | ne l1..4] per code:')
for k, v in prof(mu1-mu0).items(): print(' ', k, v)
df2 = G3[H100[0]].copy(); m = df2.event.isin(['insulin','norepinephrine']); df2.loc[m,'rate'] *= 2; df2.loc[m,'dose'] *= 2
Bx = bundles_of_stay(df2, RU0); Ux = np.array([b['u'] for b in Bx if b['nA']>0]); Xx = lnp(Ux, n1w, n1b)
print('x2 decode ins', np.round(Xx@w_ins,2)[:8], 'true', df2[df2.event=='insulin'].rate.values[:8]); print('x2 decode ne', np.round(Xx@w_ne,3)[:8], 'true', np.round(df2[df2.event=='norepinephrine'].rate.values[:8],3))
```