The records support an insulin effect near −2 per rate unit and a norepinephrine effect near +3.3; neither appears absent. The saved candidates are intact, but the recent cache variables use different names than expected. I’ll locate them and recover the target calculations so the next edit is based on verified evidence.

```python
r214_items = [(k, v) for k, v in list(globals().items())
              if not k.startswith("__")]
print("=== MOST RECENT VARIABLES ===")
for name, v in r214_items[-110:]:
    if isinstance(v, dict):
        print(name, "dict", list(v)[:10])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(name, tuple(v.shape), str(v.dtype))
    elif callable(v):
        try:
            print(name, str(inspect.signature(v)))
        except Exception:
            print(name, "callable")
    elif hasattr(v, "shape"):
        print(name, type(v).__name__, v.shape)
    else:
        print(name, type(v).__name__, repr(v)[:90])

print("=== CAUSAL TARGET NOTES ===")
for section in Path("/work/notes.md").read_text().split("\n## "):
    header = section.split("\n", 1)[0]
    if header.startswith(("Run 4\n", "Run 5", "Run 6")) or any(
        t in section.lower() for t in ["target shift", "causal target", "target effect",
                                      "0.78", "persistence estimate"]):
        print(section[:1600])
        if len(section) > 1600:
            print("[section truncated]")
        if "r214_target_limit" not in globals():
            r214_target_limit = 0
        r214_target_limit += 1
        if r214_target_limit >= 4:
            break
```