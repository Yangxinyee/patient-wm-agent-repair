The tilt directions are there: `tg` and `tm` are logit directions over bins 1 to 16, ordered by bin centre, for glucose and MAP. `gvec` is a unit vector, presumably the gating direction that separates codes.

My comparison failed because `make_edit` does not return the same structure as `ED`. I will print what it returns, with keys and types one level down, and then repeat the comparison on the right objects.

```python
E1=make_edit(1.0,1.0)
print(type(E1).__name__, (list(E1.keys()) if isinstance(E1,dict) else len(E1)))
if isinstance(E1,dict):
    for k,v in E1.items():
        print(' ',k,type(v).__name__, (tuple(v.shape) if hasattr(v,'shape') else (list(v.keys())[:8] if isinstance(v,dict) else str(v)[:80])))
```