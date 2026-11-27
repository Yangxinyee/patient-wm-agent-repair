Found the cause. My local edited weights `WE` include a change to `core.prelude.0.attn.kv.weight`, but that tensor was never saved as an edit. It looks like the stage-1 rank-1 value edit from the earlier design, which feeds the dose signal into the slots.

If the n3 decoders for (rate − c_hat) were fitted with that kv change in place, then in the tool, which lacks it, they read a different n3. That would explain all three symptoms:
- the 2.6× larger, weakly correlated state changes in fwd_5;
- the biased residual and the upward drift;
- the missing dose response.

To confirm, I'll look at the structure of the kv difference, then rerun my local comparison (`run_cmp`) on the fwd_5 stays with `WE` but the kv tensor reset to the original.

If the edited-model error falls to the original-model level (about 1e-4) and nd_loc ≈ nd_tool, the tool is running "my edit minus kv", and the cause is confirmed. Then I will decide between saving the kv edit and refitting the decoders without it.

```python
import numpy as np, torch, time
kvn = 'core.prelude.0.attn.kv.weight'
we_kv = WE[kvn]; o_kv = load_original(kvn)
to64 = lambda t: (t.double().numpy() if hasattr(t,'double') else np.asarray(t, float))
dkv = to64(we_kv) - to64(o_kv)
print(kvn, type(we_kv).__name__, getattr(we_kv,'dtype',None), dkv.shape, 'max', np.abs(dkv).max(), 'sv', np.round(np.linalg.svd(dkv, compute_uv=False)[:4],4))
rows = np.where(np.abs(dkv).max(1)>0)[0]; cols = np.where(np.abs(dkv).max(0)>0)[0]
print('nonzero rows', len(rows), rows[:12], ' nonzero cols', len(cols))
WEt = dict(WE); WEt[kvn] = (o_kv.to(we_kv.dtype) if hasattr(we_kv,'dtype') else to64(o_kv))
t0 = time.time(); rr_new = []
try:
    out_ = run_cmp(WEt, rr_new)
    print('run_cmp time %.1f' % (time.time()-t0), type(out_).__name__, 'filled', len(rr_new))
    src_ = rr_new if len(rr_new) else (out_ if isinstance(out_, list) else None)
    if src_:
        Dn = pd.DataFrame(src_); Dn['drugc'] = Dn['drug'].astype(str)
        cols_ = [c for c in ['err_o','err_e','nd_tool','nd_loc'] if c in Dn]
        print(Dn.groupby('drugc')[cols_].median())
        if 'nd_loc' in Dn: 
            for dc,g in Dn.groupby('drugc'): print(dc, 'corr %.3f ratio %.3f' % (np.corrcoef(g.nd_tool,g.nd_loc)[0,1], np.median(g.nd_loc/np.maximum(g.nd_tool,1e-12))))
except Exception as ex:
    import traceback; traceback.print_exc()
```