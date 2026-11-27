The inspection was truncated again because a function’s default arguments included the weight dictionary. No weights changed. I’ll print only names and shapes this time; that should expose the evaluator without flooding the output, allowing a reliable factual-loss screen.

```python
r215_items = [(k, v) for k, v in list(globals().items())
              if not k.startswith("__")]
print("=== RECENT VARIABLES, COMPACT ===")
for name, v in r215_items[-140:]:
    if isinstance(v, dict):
        print(name, "dict keys", list(v)[:8])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(name, tuple(v.shape), str(v.dtype))
    elif callable(v):
        code = getattr(v, "__code__", None)
        print(name, "function args", code.co_varnames[:code.co_argcount] if code else "")
    elif hasattr(v, "shape"):
        print(name, type(v).__name__, v.shape)
    elif isinstance(v, (str, int, float, bool, type(None))):
        print(name, type(v).__name__, repr(v)[:85])
    else:
        print(name, type(v).__name__)
print("=== CAUSAL TARGET PASSAGES ===")
r215_count = 0
for section in Path("/work/notes.md").read_text().split("\n## "):
    header = section.split("\n", 1)[0]
    if header.startswith(("Run 4", "Run 5", "Run 6")) and not header.startswith(
            ("Run 40", "Run 41", "Run 42", "Run 43", "Run 44", "Run 45",
             "Run 46", "Run 47", "Run 48", "Run 49", "Run 50", "Run 51",
             "Run 52", "Run 53", "Run 54", "Run 55", "Run 56", "Run 57",
             "Run 58", "Run 59", "Run 60", "Run 61", "Run 62", "Run 63",
             "Run 64", "Run 65", "Run 66", "Run 67", "Run 68", "Run 69")):
        print(section[:1800])
        r215_count += 1
        if r215_count >= 2:
            break
```