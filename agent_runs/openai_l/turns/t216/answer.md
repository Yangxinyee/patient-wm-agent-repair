The architecture confirms that the value head has a nonlinear layer where a sensor-specific correction could be added without directly changing the other heads. I’ll inspect the existing dose decoders and likelihood helpers next. A useful candidate must preserve factual logits while producing the records-based response to dose changes; dose sensitivity alone is not enough.
```python
import dis

print("=== RECENT FIT OBJECTS ===")
for n in sorted(list(globals())):
    if n.endswith(("191", "192", "193")):
        v = globals()[n]
        if isinstance(v, dict):
            print(n, "keys", list(v)[:18])
        elif hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif callable(v):
            print(n, "args", v.__code__.co_varnames[:v.__code__.co_argcount])
        elif isinstance(v, (int, float, str, list, tuple)):
            print(n, repr(v)[:300])

print("\n=== HELPER AND DATA INVENTORY ===")
for n in sorted(list(globals())):
    v = globals()[n]
    if callable(v) and any(s in n.lower() for s in
        ["feature", "decode", "head", "logit", "cf_", "nll", "state"]):
        c = getattr(v, "__code__", None)
        if c is not None:
            print(n, c.co_varnames[:c.co_argcount])
for n in ["rows74", "rows67", "events74", "centres_train67", "eps67",
          "cf_features", "cf_features178", "cf_features180", "train", "tuning",
          "gate192", "gatepred192", "gate_y192", "YF179",
          "directions193", "fitmask179", "head_original185"]:
    if n not in globals():
        continue
    v = globals()[n]
    if isinstance(v, dict):
        print(n, "keys", list(v)[:20])
        for k in list(v)[:2]:
            u = v[k]
            print(" ", k, "shape", getattr(u, "shape", None),
                  "keys", list(u)[:12] if isinstance(u, dict) else "")
    elif hasattr(v, "columns"):
        print(n, v.shape, list(v.columns), "\n", v.head(2).to_string(index=False))
    elif hasattr(v, "shape"):
        a = arr192(v)
        print(n, a.shape, "range", float(np.nanmin(a)), float(np.nanmax(a)))
    else:
        print(n, type(v).__name__)

print("\n=== FIT DETAILS ===")
for n in ["target_logits83", "head_inputs96", "decoder_base180"]:
    print("\n", n)
    dis.dis(globals()[n])
print("\n=== RECORD-EFFECT NOTES ===")
print(Path("/work/notes.md").read_text()[:6500])
```
