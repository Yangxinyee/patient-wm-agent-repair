The trace shows how `run_variant3` works. For each bundle it gets vectors from `bvd`, times in seconds, and memory = the last 128 non-skipped bundle vectors with their ages. It then calls `step(out_prev, h_prev, dt, e, u, z, Mem, ages)` using the tool's posterior z. For counterfactual doses I must compute z myself through `z_post` (from ctx, e, u), because the tool's z only fits recorded doses. Next I'll look at one bundle's structure and what `bvd` returns, and find the `gap` weights needed for ctx.

```python
st_=list(BUN.keys())[0]; b1=BUN[st_][1]
print('bundle type', type(b1).__name__, (list(b1.keys()) if isinstance(b1,dict) else (len(b1), [type(x).__name__ for x in b1])))
print('bundle repr', str(b1)[:300])
ov=bvd(b1); print('bvd ->', type(ov).__name__, [tuple(getattr(x,'shape',())) if hasattr(x,'shape') else type(x).__name__ for x in (ov if isinstance(ov,(tuple,list)) else [ov])])
print([n for n in NAMES if 'gap' in n or 'core.age' in n or 'type_emb' in n or 'query' in n][:12])
```