The FFN fit has usable dose contrast, especially for norepinephrine. I’ll now compare two possible injection points: the FFN output and the direct recurrent input. A useful point must change the reconstructed next-reading prediction without requiring a large disturbance on factual bundles.

```python
print("=== LOCAL HELPERS ===")
fn_names116 = []
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            sig116 = str(inspect.signature(v))
        except Exception:
            sig116 = ""
        fn_names116.append((k, sig116))
print("\n".join(f"{k}{s}" for k, s in fn_names116))

print("\n=== LOCAL STEP DATA ===")
for k in sorted(list(globals())):
    if k.endswith("110") and k not in [
        "latent_J110", "latent_out_J110", "rr110",
        "base_head110", "probe_head110"
    ]:
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "keys", list(v)[:12])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v),
                  [type(a).__name__ for a in v[:3]])
        elif not callable(v):
            print(k, repr(v)[:140])

print("\n=== FIT SHAPES ===")
for family116 in ["ffn", "E"]:
    fit116 = prelude_fits112[family116][0.3]
    print(family116, {k: getattr(v, "shape", None)
                      for k, v in fit116.items() if k != "report"})

print("\n=== CONTEXT HELPER DEPENDENCIES ===")
for k in ["previous_context_repair", "prelude_features111", "lin104",
          "norm104", "layer104"]:
    fn116 = globals()[k]
    print(k, fn116.__code__.co_names)
    print("constants:", [v for v in fn116.__code__.co_consts
                          if isinstance(v, str)])
print("\nHead configuration:", repair_head_cfg)
print("History available:", [k for k in ["_ih", "In", "history"] if k in globals()])
```
