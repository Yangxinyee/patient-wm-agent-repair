Confirmed: b1 and b2 were overwritten by stale variables. W1 and W2 still have the right shapes but may be stale too. I reload all five value-head tensors from the original weights, reporting whether W1, W2 and TR had drifted, then re-run the planned measurement unchanged:
- the lag profile of the head-2 own-direction injection, for both drugs and all value codes;
- linear decoding of doubled doses.

This run is what tells me whether amplifying the model's own direction acts mainly at lag 1, or persists and leaks across drugs.

```python
for nm, tn in [('W1','attr_heads.value_bin.0.weight'),('b1','attr_heads.value_bin.0.bias'),('W2','attr_heads.value_bin.2.weight'),('b2','attr_heads.value_bin.2.bias'),('TR','embed_transition.value_bin.weight')]:
    new = load_original(tn).numpy() if hasattr(load_original(tn),'numpy') else np.asarray(load_original(tn))
    old = globals()[nm]; same = (np.shape(old)==new.shape) and np.allclose(old, new)
    globals()[nm] = new.astype(np.float64); print(nm, 'was same' if same else 'RELOADED')
t0 = time.time(); rows0, mu0 = run_rows(H100); print('rows', len(rows0), 'code ex', rows0[0][:3], 'secs %.1f' % (time.time()-t0))
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