The cached probe directions and gate are available, and no additional tensors were changed. The inventory was truncated because function defaults contained tensors. I’ll print only compact metadata and the causal-check summary, then use those caches for a local factual-loss test.

```python
snapshot135 = list(globals().items())
print("=== HELPER NAMES AND PARAMETER NAMES ONLY ===")
for k, v in snapshot135:
    if callable(v) and getattr(v, "__module__", None) == "__main__":
        try:
            print(k + "(" + ", ".join(inspect.signature(v).parameters) + ")")
        except (TypeError, ValueError):
            print(k)

print("=== RELEVANT VARIABLE NAMES ===")
for prefix135 in ("r10", "r11", "cache", "meta", "qq", "zz", "gate", "feat", "causal"):
    print(prefix135, [k for k, v in snapshot135 if k.startswith(prefix135)])

print("=== SELECTED CACHE CONTENTS ===")
for key135 in ("cache98", "meta98", "r132_feat", "r132_candidates"):
    obj135 = globals().get(key135)
    if isinstance(obj135, dict):
        print(key135)
        for sub135, val135 in obj135.items():
            if hasattr(val135, "shape"):
                print(" ", sub135, type(val135).__name__, tuple(val135.shape))
            elif isinstance(val135, dict):
                print(" ", sub135, "dict", list(val135.keys()))
            else:
                print(" ", sub135, type(val135).__name__, str(val135)[:100])
    else:
        print(key135, type(obj135).__name__)

notes135 = Path("/work/notes.md").read_text()
a135 = notes135.find("## Run 4\n")
b135 = notes135.find("\n## ", a135 + 5)
print("=== ACCEPTED EARLY CAUSAL CHECK ===")
print(notes135[a135:b135][:3000])
print("=== EDIT HASH INPUT SHAPES ===")
for name135 in edited():
    print(name135, tuple(load(name135).shape))
```
