The records support a negative insulin effect on glucose and a positive norepinephrine effect on MAP, while the original rollouts respond only weakly. The architecture confirms that a repair must reach the value head and survive feedback through sampled readings. I’ll recover the earlier robustness checks and available helpers; inconsistent estimates or poor signal persistence would rule out the current approach.

```python
print("=== EARLIER CAUSAL AND VALIDATION NOTES ===")
print(notes130[6000:12500])

print("=== HELPER NAMES ===")
helper_names131 = [
    k for k, v in list(globals().items())
    if inspect.isfunction(v) and not k.startswith("_")
]
print(", ".join(sorted(helper_names131)))

print("=== RECENT OBJECT INVENTORY ===")
recent131 = [
    (k, v) for k, v in list(globals().items())
    if k.endswith(("117", "118", "123", "125", "127", "128", "129"))
    and not inspect.isfunction(v)
]
for k131, v131 in sorted(recent131):
    if isinstance(v131, dict):
        print(k131, "dict", list(v131.keys())[:20])
    elif hasattr(v131, "shape"):
        print(k131, tuple(v131.shape))
    elif isinstance(v131, (str, int, float, bool, list, tuple)):
        print(k131, repr(v131)[:110])
```
