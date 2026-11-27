The architecture confirms that a correction must survive recurrence and reach the shared value head; changing the latent posterior alone is insufficient. The output was truncated before showing the saved helpers, so I’ll inspect a compact inventory next. This should reveal whether the private-state route can be implemented and tested directly.

```python
import inspect, re

func120 = [(k, v) for k, v in list(globals().items())
           if callable(v) and getattr(v, "__module__", None) == "__main__"]
print("=== FUNCTION NAMES ===")
print(" ".join(k for k, v in func120))
print("\n=== RECENT FUNCTION SIGNATURES ===")
for k120, v120 in func120[-35:]:
    try:
        print(k120, inspect.signature(v120))
    except Exception:
        print(k120)

print("\n=== RECENT VARIABLES ===")
vars120 = [(k, v) for k, v in list(globals().items())
           if not k.startswith("_") and not callable(v)]
for k120, v120 in vars120[-115:]:
    if isinstance(v120, (np.ndarray, torch.Tensor)):
        desc120 = f"shape={tuple(v120.shape)}, dtype={v120.dtype}"
    elif isinstance(v120, dict):
        desc120 = f"dict keys={list(v120)[:9]}"
    elif isinstance(v120, (list, tuple)):
        desc120 = f"{type(v120).__name__}, length={len(v120)}"
    elif isinstance(v120, (int, float, str)):
        desc120 = repr(v120)[:100]
    else:
        desc120 = type(v120).__name__
    print(k120, desc120)

print("\n=== CAUSAL NOTEBOOK SUMMARY LINES ===")
lines120 = open("/work/notes.md").read().splitlines()
hits120 = [s for s in lines120
           if re.search(r"accepted|impulse|kernel|causal target|−1\.94|-1\.94|true effect", s, re.I)]
for s120 in hits120[-12:]:
    print(s120[:350])
print("\nCurrent edits:", edited())
```