The cached inputs include the same prior noise used by factual scoring, so candidate screening can account for that noise rather than treating the head as deterministic. I’ll build separate halving and doubling constraints and compare their factual cost. A doubling-focused direction must generalize to held-out stays—not merely fit the training contrasts—to justify a forward test.

```python
print("=== CONTRAST STRUCTURES ===")
for name in ["cf146", "cf172", "wm_train145", "wm_tune145"]:
    if name not in globals():
        continue
    obj = globals()[name]
    print(name, "keys", list(obj)[:20])
    for k, v in list(obj.items())[:8]:
        if isinstance(v, dict):
            print(" ", k, [(str(a), getattr(b, "shape", type(b).__name__))
                           for a, b in list(v.items())[:14]])
        else:
            print(" ", k, getattr(v, "shape", type(v).__name__))

print("\nFitting/CV readings:", int(cal190_fitmask.sum()), int(cal190_cvmask.sum()))
print("head177_eval defaults:", head177_eval.__defaults__)
print("MAP current frontier:")
print(pd.read_csv("/work/map_calibrated_frontier.csv").to_string(index=False)[:1700])
print("\n=== CONTRAST METADATA VARIABLES ===")
for k, v in list(globals().items()):
    if re.search(r"(meta|event|centres|dr|active|split|fitmask|cv_mask)", k, re.I):
        if isinstance(v, pd.DataFrame) and len(v) in [4510, 4751]:
            print(k, v.shape, list(v.columns))
        elif isinstance(v, (np.ndarray, torch.Tensor)) and (
            getattr(v, "shape", None) in [(4510,), (4510, 2), (4751,)]):
            print(k, tuple(v.shape), getattr(v, "dtype", None))
print("\n=== HEAD-EVALUATOR RETURN ===")
audit203_base = head177_eval(wm_train145, head177_compressed)
print([(k, getattr(v, "shape", type(v).__name__)) for k, v in audit203_base.items()])
```
