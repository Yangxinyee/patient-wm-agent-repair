The decoder exposes a 192-dimensional feature vector before its final linear layer, so a head-only regression edit is possible. The record-derived aggregate targets are roughly −11 to −13 for doubled insulin and +4 to +4.5 for doubled norepinephrine, depending on weighting. I’ll align factual and altered-dose features at the next reading; their factual reconstruction must match before I fit anything.

```python
print("=== DATA AND FEATURE OBJECTS ===")
for n in ["rows67", "validrows73", "pi73", "dtread73", "ctxread73",
          "epsread73", "hidden73", "logits73", "events74",
          "rates_by_state73", "owners72", "counts72", "drugs65",
          "rates65", "doses65", "drug_indices49", "state_train_ids36",
          "sx63", "sid63", "hour63", "e_valid60", "u_valid60",
          "const_train67", "trans_train67", "centres_train67"]:
    v = globals().get(n)
    print(n, type(v).__name__, getattr(v, "shape", None))
    if hasattr(v, "columns"):
        print(" columns:", list(v.columns))
        print(v.head(2).to_string(index=False))
    elif isinstance(v, dict):
        print(" keys:", list(v)[:8])
    elif isinstance(v, (list, np.ndarray)) and np.ndim(v) <= 1:
        print(" first:", repr(v[:6])[:250])

print("\n=== BUNDLE INPUT HELPER ===")
print(inspect.signature(bundle_inputs74))
for ins in dis.get_instructions(bundle_inputs74):
    if ins.opname not in ["RESUME", "CACHE", "PRECALL", "PUSH_NULL"]:
        print(ins.opname, ins.argrepr)

print("\n=== VALUE HEAD SHAPES ===")
for n in NAMES:
    if "attr_heads.value_bin" in n:
        print(n, SHAPES[n])

print("\n=== CROSS-TREATMENT EFFECT STRUCTURE ===")
print(repr(globals().get("effects15"))[:1700])
```