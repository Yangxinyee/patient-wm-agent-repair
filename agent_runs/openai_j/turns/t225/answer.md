The later record-based estimates put norepinephrine’s doubling effect near +3.8 and halving near −1.9, although their identification assumptions are stronger than insulin’s. Our insulin shifts are close to their targets, but norepinephrine doubling remains too small. Factual loss rose by 0.00345 nats, within the 0.005 limit. I’ll inspect the head-edit construction for a more selective correction; stronger doubling response without exhausting that margin would confirm its usefulness.

```python
import inspect

print("=== RECONSTRUCTION AND REPAIR NOTES ===")
for i in [39, 47, 49, 54, 56, 57, 60]:
    print(sections[i][:1050])

print("\n=== HEAD SHAPES AND HELPERS ===")
for n in NAMES:
    if n.startswith("attr_heads.value_bin"):
        print(n, SHAPES[n])
print("head177_make globals:", head177_make.__code__.co_names)
try:
    print(inspect.getsource(head177_make)[:3000])
except (OSError, TypeError):
    import dis, io
    buf = io.StringIO()
    dis.dis(head177_make, file=buf)
    print(buf.getvalue()[:2500])

v = map192_candidates[(10, 6)]
print("Current MAP coefficients:", type(v).__name__,
      getattr(v, "shape", None), str(v)[:500])
print("\n=== RELEVANT PERSISTENT OBJECTS ===")
for k, v in list(globals().items()):
    if re.search(r"(177|192|map16|map17|head14|head15|head16|raw|contrast)", k, re.I):
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns)[:10])
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys", str(list(v)[:6])[:180])
        elif callable(v):
            print(k, "function", v.__code__.co_varnames[:v.__code__.co_argcount])
print("\n=== SAVED CANDIDATE KEYS ===")
for fn in ["best_combined_v1.npz", "best_insulin_v1.npz",
           "validated_combined_backup.npz", "map_cal_candidate_v2.npz"]:
    zf = np.load("/work/" + fn)
    print(fn, [(k, zf[k].shape) for k in zf.files])
```
