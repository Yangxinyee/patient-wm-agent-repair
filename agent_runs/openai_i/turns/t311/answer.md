The zero-response route already used almost the entire factual-loss allowance, so reducing drug-response strength alone may not solve the problem. I’ll inspect the compensated head and test whether routing through a less disruptive head is feasible. Lower held-out reconstruction error would support that approach; unchanged or worse error would argue against it.

```python
print("=== ROUTE GEOMETRY ===")
print("response_cols228:", response_cols228)
print("directions231:", {k: np.shape(v) for k, v in directions231.items()})
print("\n=== SMALL-ROUTE FIT VARIABLES ===")
for k, v in list(globals().items()):
    if k.endswith("222"):
        if hasattr(v, "shape"):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "keys", list(v)[:15])
        elif isinstance(v, (list, tuple, int, float)):
            print(k, repr(v)[:250])
print("\n=== ATTENTION FEATURE HELPER ===")
try:
    print(inspect.getsource(att_features220)[:4600])
except (OSError, TypeError):
    print("signature", inspect.signature(att_features220))
    print("names", att_features220.__code__.co_names)
    print("constants", repr(att_features220.__code__.co_consts)[:650])
    instructions273 = list(dis.get_instructions(att_features220))
    print("\n".join(f"{x.opname} {x.argrepr}" for x in instructions273)[:4600])
print("\n=== ORIGINAL/TOKEN SAMPLE CONTRACT ===")
print("tokens220", tokens220.shape, "Hroute220", Hroute220.shape,
      "Yorig220", Yorig220.shape, "Yroute220", Yroute220.shape)
print("args_cache182 example:", type(args_cache182[0]),
      [(type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else repr(x)[:80])
       for x in args_cache182[0]])
print("\n=== RESPONSE FACTORY ===")
print("names", response_candidate231.__code__.co_names)
print("constants", response_candidate231.__code__.co_consts)
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 273 — alternative-head compensation plan\n"
            "- Small zero-response route had dNLL +.00509, so routing overhead needs reduction independently of response gain.\n"
            "- Inspecting reusable attention/FFN features to compare compensated replacement of other heads.\n"
            "- No weights changed.\n")
```