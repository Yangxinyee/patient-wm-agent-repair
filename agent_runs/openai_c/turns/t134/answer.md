The architecture confirms that doses enter through action tokens before the recurrent update. The recovered insulin threshold estimates are substantially more negative than the original model’s response, but the output cut off before the remaining evidence. I’ll recover that evidence compactly so the next test is judged against record-based targets, not just a stronger-looking response.

```python
import re

print("NOTEBOOK LENGTH", len(notes_text))
print("\nEARLY EVIDENCE, CONTINUED")
print(notes_text[4500:11000])

print("\nEXPERIMENT HEADINGS")
for line in notes_text.splitlines():
    if line.startswith("##"):
        print(line)

print("\nCURRENT OPERATIONS", ops())
print("\nRELEVANT SAVED VARIABLES")
for key, val in list(globals().items()):
    if key.startswith("_"):
        continue
    if any(s in key.lower() for s in
           ["causal", "target", "effect", "ridge", "sensor", "direction", "probe", "panel", "fit", "gain"]):
        if isinstance(val, np.ndarray):
            print(key, "array", val.shape)
        elif isinstance(val, pd.DataFrame):
            print(key, "DataFrame", val.shape)
        elif isinstance(val, dict):
            print(key, "keys", list(val)[:10])
        elif isinstance(val, (float, int, str)):
            print(key, repr(val)[:100])
        elif isinstance(val, (list, tuple)):
            print(key, repr(val)[:180])
        elif callable(val):
            import inspect
            print(key, str(inspect.signature(val)))
```
