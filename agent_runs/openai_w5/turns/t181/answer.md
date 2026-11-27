The architecture confirms that sampled readings feed back into later predictions. The output was truncated before the identification results and helper details, so I’ll retrieve compact summaries. A correct direct effect can still shrink or become asymmetric if that feedback pathway is wrong.

```python
print("=== EARLY NOTEBOOK: RECORD-DERIVED EFFECTS ===")
print(Path("/work/notes.md").read_text()[:6500])

print("=== FUNCTION SIGNATURES ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("=== KEY OBJECT NAMES ===")
keys157 = [n for n in globals() if any(
    s in n.lower() for s in ["clean", "channel", "pair", "coef", "theta", "effect", "rollout",
                              "proxy", "spec", "cache", "rates", "ids", "transport"]
)]
print(keys157)

print("=== CURRENT HEAD SPECS ===")
for s in _r148_specs:
    print({k: s[k] for k in ["event", "drug", "beta", "c"] if k in s},
          "keys", list(s.keys()))
print("=== FORWARD FILES ===")
for d in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[1])):
    print(d.name, [(p.name, p.stat().st_size) for p in d.iterdir()])
```
