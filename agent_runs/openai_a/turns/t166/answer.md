The architecture confirms that a value-head edit can change dose responses, but it can also disturb reading-history feedback. The inventory was truncated, so I’ll retrieve it compactly before fitting anything. I need both synthetic dose contrasts and factual protection features to proceed safely.

```python
import inspect, numpy as np, pandas as pd
lines = []
items = list(globals().items())
for key, obj in items:
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        lines.append(f"FUNCTION {key}{inspect.signature(obj)}")
lines.append("RECENT VARIABLES")
for key, obj in items[-100:]:
    if key.startswith("_"):
        continue
    if isinstance(obj, np.ndarray):
        lines.append(f"{key}: array {obj.shape} {obj.dtype}")
    elif isinstance(obj, pd.DataFrame):
        lines.append(f"{key}: dataframe {obj.shape}")
    elif isinstance(obj, dict):
        lines.append(f"{key}: dict keys={list(obj.keys())[:12]}")
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{key}: {type(obj).__name__} len={len(obj)}")
    elif isinstance(obj, (str, int, float, bool, np.number)):
        lines.append(f"{key}: {str(obj)[:160]}")
print("\n".join(lines)[:8500])
print("\nNOTES END\n" + open("/work/notes.md").read()[-1100:])
```
