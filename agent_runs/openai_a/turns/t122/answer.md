The failed test included admission events in some measurement bundles, so its overall error does not isolate the measurement encoding. I’ll inspect the working action encoder and separate the mismatch by bundle. If ordinary measurement-only bundles also fail, I’ll focus on their attributes rather than admission handling.

```python
print("=== ACTION ENCODER ===")
try:
    print(inspect.getsource(rm_insulin_atoms))
except Exception:
    print("\n".join(f"{i.opname} {i.argrepr}"
                    for i in dis.get_instructions(rm_insulin_atoms)
                    if i.opname not in ("CACHE", "RESUME", "PRECALL")))

print("\n=== SET ENCODER ===")
try:
    print(inspect.getsource(rm_set))
except Exception:
    print("\n".join(f"{i.opname} {i.argrepr}"
                    for i in dis.get_instructions(rm_set)
                    if i.opname not in ("CACHE", "RESUME", "PRECALL")))

print("\n=== EXACT-MATCH RESULT VARIABLES ===")
for k, v in sorted(dict(globals()).items()):
    if "96" in k and isinstance(v, (dict, list, tuple)):
        if isinstance(v, dict):
            print(k, list(v.keys())[:8])
            for kk in ("spec", "best_spec", "rmse", "best", "score"):
                if kk in v:
                    print(kk, repr(v[kk])[:200])
        elif isinstance(v, list):
            print(k, "length", len(v), repr(v[:2])[:300])
        else:
            print(k, repr(v)[:300])

print("\n=== MEASUREMENT ERRORS BY BUNDLE ===")
_per_bundle = ((r103_best["z"] - r103_target).square().mean(1).sqrt()).numpy()
for j in range(min(16, len(r103_bundle_rows))):
    b = r103_bundle_rows[j]
    print(int(b.stay_id.iloc[0]), float(b.hour.iloc[0]),
          list(b.event), float(_per_bundle[j]))

print("\n=== STATE FILES ===")
for p in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    a = np.load(p)
    print(str(p), [(k, a[k].shape) for k in a.files])
```
