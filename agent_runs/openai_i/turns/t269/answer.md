The architecture confirms that the edited response passes through the recurrent state, so leakage can affect later predictions too. The output was truncated before showing the weight sizes. I’ll retrieve those compactly; very large response columns would support the amplification hypothesis.

```python
import inspect

print("=== RESPONSE DIRECTION SCALES ===")
z240 = np.load("/work/response_directions231.npz")
for key240 in z240.files:
    a240 = z240[key240]
    if np.issubdtype(a240.dtype, np.number):
        print(key240, a240.shape, "norm", float(np.linalg.norm(a240)),
              "max", float(np.max(np.abs(a240))) if a240.size else 0)

print("\n=== ROUTE TENSOR NAMES ===")
z240r = np.load("/work/small_route222.npz")
print([(k, z240r[k].shape) for k in z240r.files])

print("\n=== REPLAY / ROUTING OBJECTS ===")
for name240 in sorted(list(globals())):
    if any(name240.endswith(str(k)) for k in [209, 214, 222, 228, 230, 231, 237, 238]):
        obj240 = globals()[name240]
        if callable(obj240):
            print(name240, str(inspect.signature(obj240)),
                  "uses", obj240.__code__.co_names[:22])
        elif isinstance(obj240, dict):
            print(name240, "dict", list(obj240)[:8])
        elif hasattr(obj240, "shape"):
            print(name240, tuple(obj240.shape))
        elif isinstance(obj240, (tuple, list)):
            print(name240, type(obj240).__name__, len(obj240))

print("\n=== CURRENT EDITS ===")
print(edited())

with open("/work/notes.md", "a") as f240:
    f240.write("\n## Python 240 — compact recovery after truncated output\n"
               "- Architecture confirms latent action transport through a recurrent prelude.\n"
               "- Inspecting response norms and available replay helpers; no weights changed.\n")
```
