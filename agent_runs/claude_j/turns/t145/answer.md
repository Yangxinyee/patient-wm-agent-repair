A bundle is (minute, list of atoms), and each atom is (event, source, record row). So `bvd` takes the atom list. To build counterfactual doses I can copy a record row with `_replace(rate=…, dose=…)` and re-encode. Next I'll check what `bvd(atoms)` returns, look at a drug bundle's atoms, and print the `gap` / `type_emb` tensor names.

```python
ov=bvd(b1[1]); print('bvd ->', type(ov).__name__, [tuple(x.shape) if hasattr(x,'shape') else type(x).__name__ for x in (ov if isinstance(ov,(tuple,list)) else [ov])])
bd=[b for b in BUN[st_] if any(a[0]=='insulin' for a in b[1])][0]
print('drug bundle min', bd[0], [(a[0],a[1],getattr(a[2],'rate',None),getattr(a[2],'dose',None)) for a in bd[1]])
ov2=bvd(bd[1]); print('drug bvd', [tuple(x.shape) if hasattr(x,'shape') else str(x)[:20] for x in (ov2 if isinstance(ov2,(tuple,list)) else [ov2])])
print([n for n in NAMES if ('gap' in n) or ('core.age' in n) or ('type' in n) or ('query' in n)][:14])
```