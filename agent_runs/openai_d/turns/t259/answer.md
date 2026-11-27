The exposure audit refutes the guard idea: almost all excess loss occurs after treatment has run. I won’t add that guard. Next I’ll inspect the norepinephrine decoder and its controlled state probe to see whether a lower-noise direction can preserve dose sensitivity. That approach succeeds only if it improves held-out factual loss and survives an actual dose rollout.

```python
print("=== NE CONSTRUCTION NOTES ===")
notes223 = Path("/work/notes.md").read_text()
for heading in [
    "Runs 129", "Run 131:", "Run 132:", "Runs 137",
    "Run 139:", "Run 143:", "Run 190:"
]:
    pos = notes223.find(heading)
    if pos >= 0:
        end = notes223.find("\n## ", pos + len(heading))
        print(notes223[pos:end if end >= 0 else len(notes223)][:1600])

print("\n=== PROBE AND NE SOLVE VARIABLES ===")
keys223 = sorted([
    k for k in list(globals())
    if k.startswith(("r125", "r129", "r130", "r132", "r137", "r138"))
])
for k in keys223:
    v = globals()[k]
    if callable(v) and hasattr(v, "__code__"):
        print(k, "FUNCTION", inspect.signature(v), v.__code__.co_names)
    elif isinstance(v, dict):
        print(k, {
            str(a): (tuple(b.shape) if hasattr(b, "shape")
                     else list(b.keys())[:10] if isinstance(b, dict)
                     else type(b).__name__)
            for a, b in list(v.items())[:15]
        })
    elif hasattr(v, "shape"):
        print(k, tuple(v.shape), getattr(v, "dtype", ""))
    elif isinstance(v, (float, int, str, list, tuple)):
        print(k, repr(v)[:350])

print("\n=== EXACT NE PAIR ===")
ww223 = c207_tensor_weights(c206_candidates["combined_strong"])
pair223 = [48, 171]
r223_ne_v = (
    ww223["attr_heads.value_bin.0.weight"][48]
    - ww223["attr_heads.value_bin.0.weight"][171]
) / 2
r223_ne_b = float((
    ww223["attr_heads.value_bin.0.bias"][48]
    - ww223["attr_heads.value_bin.0.bias"][171]
) / 2)
r223_ne_common_v = (
    ww223["attr_heads.value_bin.0.weight"][48]
    + ww223["attr_heads.value_bin.0.weight"][171]
) / 2
r223_ne_common_b = float((
    ww223["attr_heads.value_bin.0.bias"][48]
    + ww223["attr_heads.value_bin.0.bias"][171]
) / 2)
print("difference vector norm / bias:", float(r223_ne_v.norm()), r223_ne_b)
print("common vector norm / bias:", float(r223_ne_common_v.norm()), r223_ne_common_b)
print("output columns:")
print(ww223["attr_heads.value_bin.2.weight"][:, pair223].numpy().round(5))

print("\n=== FRESH-INPUT / OLD-PROBE DIMENSIONS ===")
for k in ["r125_original", "r125_probe", "r129_x0", "r129_x1",
          "r124_x", "r170_x", "r176_x", "r132_candidates"]:
    if k in globals():
        v = globals()[k]
        if isinstance(v, dict):
            print(k, {a: tuple(b.shape) if hasattr(b, "shape") else type(b).__name__
                      for a, b in list(v.items())[:8]})
        elif hasattr(v, "shape"):
            print(k, tuple(v.shape))
```
