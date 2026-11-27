There's no local value-head implementation yet, and the factual CSV has per-reading predicted means I can validate against. Next I'll inspect what `stay_bundles` and `enc_bundle` return for one tuning stay, so I can compute prelude attention weights (slot to action-token) for the injection design.

```python
s0=int(S4['stay_id'][0]); B=stay_bundles(s0)
print(type(B), len(B))
b=B[1] if isinstance(B,list) else None
print(type(b), (list(b.keys()) if isinstance(b,dict) else (len(b) if hasattr(b,'__len__') else b)))
r=enc_bundle(b)
print(type(r), [getattr(a,'shape',a) for a in (r if isinstance(r,(tuple,list)) else [r])][:6])
```