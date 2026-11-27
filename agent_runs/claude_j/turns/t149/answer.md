`encode` calls `ln(x, prefix)`, the same convention as `lnp(x, prefix)` (LayerNorm with `W[prefix+'.weight']` / `W[prefix+'.bias']`). So I'll restore `ln = lnp`. To confirm the restoration is exact, I'll rerun the replica's one-step check against the tool's states on a few bundles of one stay. Errors near 1e-6, as before, confirm it. Then I'll look at what `bvd` returns for a reading bundle and a drug bundle.

```python
ln=lnp
errs_=[one_step_err(st_,k_) for k_ in [1,2,3,5,8]]
print('one_step_err', errs_)
ov=bvd(b1[1]); print('bvd reading ->', type(ov).__name__, [tuple(x.shape) if hasattr(x,'shape') else str(x)[:15] for x in (ov if isinstance(ov,(tuple,list)) else [ov])])
ov2=bvd(bd[1]) if 'bd' in globals() else None
bd=[b for b in BUN[st_] if any(a[0]=='insulin' for a in b[1])][0]; ov2=bvd(bd[1])
print('drug bundle min', bd[0], [(a[0],a[1],getattr(a[2],'rate',None)) for a in bd[1]][:3], '->', [tuple(x.shape) if hasattr(x,'shape') else str(x)[:15] for x in ov2])
```