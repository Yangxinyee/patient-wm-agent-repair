The architecture confirms that the current edit changes only reading predictions, not the recurrent dynamics. The output was truncated before the cached helpers appeared. I’ll retrieve a compact inventory so the next candidate uses a verified state reconstruction rather than a guessed one.

```python
import inspect
import re

snapshot134 = list(globals().items())
print("=== RECENT CACHE INVENTORY ===")
for k, v in snapshot134:
    if re.match(r"r(?:12[0-9]|13[0-3]|115|98|99)_", k):
        if hasattr(v, "shape"):
            desc = f"{type(v).__name__}{tuple(v.shape)}"
        elif isinstance(v, dict):
            desc = "dict keys=" + repr(list(v.keys())[:12])
        elif isinstance(v, (tuple, list)):
            desc = f"{type(v).__name__}, len={len(v)}"
        elif callable(v):
            try:
                desc = str(inspect.signature(v))
            except (TypeError, ValueError):
                desc = "callable"
        else:
            desc = repr(v)[:100]
        print(k, desc)

print("=== CUSTOM HELPERS ===")
for k, v in snapshot134:
    if callable(v) and getattr(v, "__module__", None) == "__main__":
        try:
            print(k, inspect.signature(v))
        except (TypeError, ValueError):
            print(k)

notes134 = Path("/work/notes.md").read_text()
print("=== CAUSAL CHECKS, EARLY NOTEBOOK ===")
start134 = notes134.find("## Run 4\n")
print(notes134[start134:start134 + 2700])
print("=== CURRENT EDITS ===", edited())
```
