The weight check settles it, and it is explanation 1, not a feedback loop:
- On every bundle without a drug, dz18 is exactly −1.50 and dz28 is −1.02. This holds before the first insulin dose as well, with sd 0.
- So the repurposed units output 0 there. My cancelling bias assumed they would output gelu(5) ≈ 5.
- That means the real action encoding u for an empty bundle is **not** zero, despite the doc. Its projection onto my decoders is strongly negative (p·u ≤ about −9).
- On drug bundles the decoder also misreads the dose badly. Rate 1–1.5 gives ≈0.2 and rate 2 gives ≈3. A norepinephrine-only bundle reads as −1.9 insulin.
- The cause is that my offline rebuild of u is inexact. The decoder norm (|p| = 379) amplifies any small error.
- The state change is small for now (|Δout| ≈ 0.02 vs 14), but the edit as saved is not neutral.

Next I'll find my rebuild functions in the kernel and test them against what the tool actually computed. From dz I can recover the true projection on every drug bundle: gelu(5 + p·u) = (dz + bias)/lam. That gives a very sensitive check on the rebuild of u, including the empty-set case.

If inverted projections match my rebuild after one fix (empty set = encoder applied to zero atoms, or a missing atom type), I can refit the decoder offline. If they disagree in an unstructured way, I'll need diagnostic passes to read u exactly.

```python
names=list(globals().keys()); i0=names.index('serf')+1
lst=[]
for k in names[i0:]:
    v=globals()[k]
    if k.startswith('_') or isinstance(v,types.ModuleType): continue
    if callable(v) and not hasattr(v,'shape'):
        try: lst.append(k+str(inspect.signature(v)))
        except Exception: lst.append(k+'()')
    elif hasattr(v,'shape'): lst.append(k+str(tuple(v.shape)))
    elif isinstance(v,(list,dict,tuple)): lst.append(k+':'+type(v).__name__+str(len(v)))
    else: lst.append(k+'='+repr(v)[:20])
print(' | '.join(lst)[:5000])
```