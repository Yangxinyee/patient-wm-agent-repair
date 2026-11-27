The architecture shows that the candidate uses the posterior and attention pathway to pass dose information into the state. The truncated output did not include the causal estimates or validation details. I’ll retrieve those selectively; both the record-based target and the candidate’s factual disturbance need to be clear before saving an edit.

```python
import inspect
from pathlib import Path

print("--- initial record evidence ---")
notes105 = Path("/work/notes.md").read_text()
print(notes105[:6500])

print("\n--- relevant retained names ---")
names106 = [
    n for n in globals()
    if not n.startswith("_") and any(
        s in n.lower() for s in (
            "104", "103", "102", "95", "transition67", "pre_context",
            "predict", "head_features", "panel", "lag", "effect", "kernel"
        )
    )
]
print(" ".join(sorted(names106)))

print("\n--- callable signatures ---")
for n in sorted(names106):
    obj = globals()[n]
    if callable(obj) and hasattr(obj, "__code__"):
        print(n, inspect.signature(obj), "uses:", obj.__code__.co_names[:28])

print("\n--- candidate tensors ---")
for n, t in candidate104.items():
    old = load_original(n)
    new = torch.as_tensor(t, dtype=torch.float32)
    print(n, tuple(new.shape),
          "change RMS", float((new-old).square().mean().sqrt()),
          "change max", float((new-old).abs().max()))

print("\n--- run-104 scalar settings ---")
for n in sorted(names106):
    obj = globals()[n]
    if isinstance(obj, (str, int, float, bool, np.number)):
        print(n, repr(obj))
```
