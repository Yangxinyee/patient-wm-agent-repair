The fresh reconstruction is stored under `rr_*`, and the notebook records adjusted insulin-negative and norepinephrine-positive associations. Those associations alone do not establish the full intervention response. I’ll first isolate the observation-encoder discrepancy; an exact posterior match would let us test candidate weight edits reliably.

```python
print("=== RR OBJECTS ===")
for name in sorted(n for n in globals() if n.startswith("rr_")):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__"):
        print(name, inspect.signature(obj), "names=", obj.__code__.co_names)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys=", repr(list(obj)[:4])[:120])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:80])
    else:
        print(name, type(obj).__name__)

print("\n=== OBSERVATION ENCODER BYTECODE ===")
dis.dis(r86_encode_observations)
print("\n=== FRESH ENCODER BYTECODE ===")
for name in sorted(n for n in globals() if n.startswith("rr_")):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__") and "sqrt" in obj.__code__.co_names:
        print(name)
        dis.dis(obj)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 108\nLocated fresh objects under rr_*. Original record evidence supports insulin-negative and NE-positive adjusted effects, subject to identification assumptions. Comparing fresh versus validated set encoders; no edits. Counts: 108 Python runs, 6 forward passes.\n")
```