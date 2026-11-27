Bundle counts match the states rows for all 100 training stays. The crash is only in my hour check: `build_bundles` evidently returns a different structure from the stored `BUN` entries, probably a dict keyed by minute rather than a list of (minute, events) pairs. I will inspect the return type and convert it to the same list-of-pairs form if needed. Then I will rerun the hour alignment and build the reading table.

```python
o4 = BUN4[selT[0]]
print(type(o4).__name__, len(o4))
if isinstance(o4, dict):
    k0_ = sorted(o4.keys())[:2]; print('keys sample', k0_, str(o4[k0_[0]])[:150])
else:
    print(str(o4[0])[:150])
def norm_bun(o):
    if isinstance(o, dict):
        return [(k, o[k]) for k in sorted(o.keys())]
    return o
BUN4 = {s: norm_bun(v) for s, v in BUN4.items()}
s_ = selT[0]; r_ = np.where(sid4 == s_)[0]
print('hour check', np.allclose(np.array([m for m, _ in BUN4[s_]]) / 60.0, S4['hour'][r_], atol=0.02))
RT4 = reading_table(S4, BUN4)
print(RT4.shape, RT4.event.value_counts().to_dict())
```