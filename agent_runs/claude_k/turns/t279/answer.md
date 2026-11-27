`enc_e_var(key, attrs)` builds a drug bundle's e from its END atoms (via `EB`) and its u from the START atoms (via `SB`), which is what I need for drug bundles. I'll check the key format of `SB`/`EB` and call it once with `attrs='none'` (the setting my notes say matches the tool). If it returns a 192-vector distinct from `e_empty()`, I'll use it for all drug bundles in the reconstruction.

```python
print(type(SB).__name__, len(SB), list(SB.keys())[:2], type(EB).__name__, len(EB), list(EB.keys())[:2])
kk=list(EB.keys())[0]
for att in ['none','full']:
    try:
        r_=enc_e_var(kk, att); r_=r_ if isinstance(r_,(tuple,list)) else (r_,)
        print(att, [np.asarray(x).shape for x in r_], [round(float(np.linalg.norm(np.asarray(x)-ee)),3) for x in r_ if np.asarray(x).shape==(192,)])
    except Exception as ex: print(att,'err',repr(ex)[:150])
```