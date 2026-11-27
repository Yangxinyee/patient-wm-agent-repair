Block 1 is the last layer of the main stack (core.blocks.0.1); after it only the coda runs, before the state is stored. The other relevant fact from my notes: the slot state (|y| ≈ 1800) dwarfs the new input E. A drug signal entering through the prelude is therefore tiny by the time it reaches the last layer. That explains why the model ignores doses, and why the residual decodes only moderately from block-1 features (R² ≈ 0.75 / 0.59).

The cleanest place to compute "rate minus policy mean, only at drug bundles" is the prelude, where the u token is strong and the state is also readable. Its output then has to be aimed through prelude_norm and the blocks toward the belief direction. Before building this I need the exact interfaces of the hooked core and helpers (cstep_h, cstep_r, prelude_full, run_tf, setW, isdrug, n3_of, tok_n1, scale) and the layout of `rows`/`feat` from run 185. I print only signatures and key lists. That tells me whether I can reuse them directly or must write thin wrappers.

```python
for nm in ['cstep_h','cstep_r','cstep_K','prelude_full','run_tf','substep','mhead','setW','resetW',
           'isdrug','sfeat','build','n3_of','tok_n1','T_post_attn','ins_atoms_scaled','scale','modread','free_run','head_eval']:
    f = globals()[nm]
    print(nm, str(inspect.signature(f))[:150])
print('feat keys', {k: (tuple(v.shape) if hasattr(v,'shape') else type(v).__name__) for k, v in feat.items()})
r0_ = rows[0]
print('rows[0]', type(r0_).__name__, (list(r0_.keys())[:12] if isinstance(r0_, dict) else (len(r0_) if hasattr(r0_,'__len__') else r0_)))
b0 = bundles_B[list(bundles_B.keys())[0]]
print('bundles_B entry', type(b0).__name__, len(b0), (list(b0[0].keys()) if isinstance(b0[0], dict) else type(b0[0]).__name__))
```