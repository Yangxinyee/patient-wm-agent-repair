The originals are restored. The notebook’s record-based estimates support roughly −2 per insulin rate unit and +3 per norepinephrine rate unit for the next reading, but not yet the cumulative rollout effects. I’ll inspect the retained channel construction to find a less disruptive route.

```python
import inspect
import torch

items206 = list(globals().items())
functions206 = [
    (k, v) for k, v in items206
    if callable(v) and getattr(v, "__module__", "") == "__main__"
]
print("RECENT FUNCTIONS:")
for k, v in functions206[-45:]:
    try:
        print(k, inspect.signature(v))
    except Exception:
        print(k)

arrays206 = [
    (k, v) for k, v in items206
    if isinstance(v, (np.ndarray, torch.Tensor))
]
print("\nRECENT ARRAYS/TENSORS:")
for k, v in arrays206[-85:]:
    print(k, tuple(v.shape), str(v.dtype))

dicts206 = [(k, v) for k, v in items206 if isinstance(v, dict)]
print("\nRECENT DICTIONARIES:")
for k, v in dicts206[-18:]:
    print(k, len(v), list(v)[:8])

print("\nRECENT SMALL VALUES:")
for k, v in items206[-100:]:
    if isinstance(v, (int, float, str)) and k not in {"notes205", "key", "name"}:
        print(k, str(v)[:120])
print("\nACTIVE EDITS:", edited())
```