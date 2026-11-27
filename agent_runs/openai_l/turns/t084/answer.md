The recovered notes show that simple state-space fits gave conflicting effects and were rejected; insulin threshold comparisons were more credible. The output was cut off before the later estimates. I’ll retrieve those estimates and the helper interfaces so the edit uses validated record evidence, not the rejected fits.

```python
import re, inspect

print("=== CAUSAL NOTE EXCERPTS ===")
sections77 = re.split(r"(?m)(?=^## )", notes76)
selected77 = []
for sec in sections77:
    head = sec.splitlines()[0] if sec.splitlines() else ""
    nums = [int(x) for x in re.findall(r"\d+", head)]
    if nums and 12 <= nums[0] <= 36:
        selected77.append(sec.strip())
text77 = "\n\n".join(selected77)
print(text77[:6100])

print("\n=== FUNCTIONS (RECENT HELPERS FIRST) ===")
recent77 = [(n, s) for n, s in funcs76
            if any(k in n.lower() for k in
                   ["63", "64", "65", "67", "72", "transition", "decode",
                    "register", "encoder", "action", "feature", "causal", "kernel"])]
for n, s in recent77:
    print(n, s)

print("\n=== POTENTIAL TARGET AND FIT VARIABLES ===")
for n, v in list(globals().items()):
    if not n.startswith("_") and any(k in n.lower() for k in
           ["kernel", "causal", "rd_", "innovation", "target", "rich_fit"]):
        if inspect.isfunction(v):
            continue
        if hasattr(v, "shape"):
            print(n, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "dict keys:", list(v)[:15])
        elif isinstance(v, (list, tuple)):
            print(n, type(v).__name__, "length", len(v), "first", repr(v[:2])[:160])
        elif isinstance(v, (float, int, str)):
            print(n, repr(v)[:180])
```