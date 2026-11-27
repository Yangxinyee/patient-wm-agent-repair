The architecture confirms that the correction must operate through the shared value head, while preserving predictions for other reading types. I will locate the activity separator, correction recipe, and saved checkpoints. A useful gate must suppress the correction during inactive hours without changing its dose sensitivity during active hours.

```python
import re
import inspect

print("=== USER-DEFINED FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\n=== RELEVANT VARIABLES ===")
pattern = re.compile(r"active|activity|gate|sep|r3|checkpoint|cache|panel|ridge|finger", re.I)
for name, obj in sorted(list(globals().items())):
    if not pattern.search(name) or name in {"obj", "pattern"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), repr(list(obj.keys())[:10])[:240])
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(name, "table", obj.shape, list(obj.columns)[:10])
    elif isinstance(obj, (str, int, float, bool, tuple)):
        print(name, repr(obj)[:240])

print("\n=== RECORD-DERIVED EFFECT NOTES ===")
lines = Path("/work/notes.md").read_text().splitlines()
hits = [i for i, s in enumerate(lines)
        if re.search(r"fuzzy.RD|persistence|NE.*(null|effect|estimate)|norepinephrine.*(null|effect)|target.*train", s, re.I)]
for i in hits[-14:]:
    print("\n".join(lines[max(0, i-1):min(len(lines), i+3)]))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Activity-gate recovery; Python 233, forwards 26/50\n"
            "- Re-read architecture and data documentation.\n"
            "- Latest activity separator has essentially perfect held-out discrimination and small dose-induced score changes.\n"
            "- This is representation evidence, not causal evidence or GPU validation. stronger6 remains installed.\n")
```
