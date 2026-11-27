The gating idea is workable. In several embedding dimensions the code embeddings separate glucose from MAP by much more than the bin embeddings vary:
- **dim 5:** MAP codes are at least 0.70 above both glucose codes, while the bin range is only 0.19. This is a natural gate for a norepinephrine→MAP unit.
- **dims 0 and 22:** glucose is at least 0.42 or 0.59 above MAP. These are natural gates for an insulin→glucose unit.

For measurement codes the rate-bin row is the zero padding row, so it adds nothing.

The common-mode contamination from LayerNorm (mean/std of h across dimensions) is 0.028 ± 0.024. With a memory of order 30 in h, that is a few percent of the signal: tolerable, and a reference dimension could cancel it later if needed.

**Next step:** obtain the prelude output E at every bundle. That is the input from which `b_in`'s new row has to decode the dose deviation, so the decoder's R² will decide how clean the memory is. `tf_states_ds` calls `step_b` through a global lookup. I will temporarily wrap `step_b` to also capture its internal E, and inspect the dataset structure.

**What success looks like:** captured E arrays aligned with bundles (stay, hour, type), so I can regress the dose signals on them in the next run.

```python
ds10 = ds_t
print({k:(tuple(v.shape) if hasattr(v,'shape') else (type(v).__name__, len(v) if hasattr(v,'__len__') else v)) for k,v in ds10.items()} if isinstance(ds10,dict) else type(ds10))
_orig_step_b = step_b
CAP = []
def step_b(hprev, e, u, z, dt, Mx, Mage, Mmask, P, ret=False):
    r1 = _orig_step_b(hprev, e, u, z, dt, Mx, Mage, Mmask, P, ret=True)
    CAP.append(r1)
    return _orig_step_b(hprev, e, u, z, dt, Mx, Mage, Mmask, P, ret=ret)
try:
    outS = tf_states_ds(P, ds10)
finally:
    step_b = _orig_step_b
print("calls", len(CAP), "ret=True type", type(CAP[0]), (list(CAP[0].keys()) if isinstance(CAP[0],dict) else len(CAP[0])))
if isinstance(CAP[0],dict):
    print({k:tuple(v.shape) for k,v in CAP[0].items() if hasattr(v,'shape')})
print("tf_states_ds returns", type(outS), [tuple(x.shape) if hasattr(x,'shape') else type(x) for x in (outS if isinstance(outS,(tuple,list)) else [outS])])
print("third tf_eval output sample", r_[2][:8])
```