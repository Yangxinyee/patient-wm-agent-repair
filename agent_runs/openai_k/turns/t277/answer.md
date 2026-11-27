The inspection encountered another metadata error; it produced no new model evidence and changed no weights. I’ll use a simpler, type-specific inventory to recover the candidate. The next substantive test remains whether stronger dose responses can be obtained without exceeding the factual NLL limit.

```python
import torch, inspect, re, numpy as np

r249_items = list(globals().items())
print("=== CANDIDATE OBJECTS ===")
for r249_n, r249_o in r249_items:
    if not re.search(r"(243|244|245)$", r249_n):
        continue
    if isinstance(r249_o, torch.Tensor):
        print(r249_n, "Tensor", tuple(r249_o.shape))
        if r249_o.numel() <= 16:
            print(r249_o.detach().cpu().numpy())
    elif isinstance(r249_o, np.ndarray):
        print(r249_n, "ndarray", r249_o.shape)
        if r249_o.size <= 16:
            print(r249_o)
    elif isinstance(r249_o, dict):
        print(r249_n, "dict", list(r249_o.keys())[:10])
    elif inspect.isfunction(r249_o):
        print(r249_n, "fn", r249_o.__code__.co_varnames[:r249_o.__code__.co_argcount])
    elif isinstance(r249_o, (list, tuple)):
        print(r249_n, type(r249_o).__name__, len(r249_o))
    elif isinstance(r249_o, (str, float, int)):
        print(r249_n, repr(r249_o)[:100])
    else:
        print(r249_n, type(r249_o).__name__)

print("\n=== REPLAY FUNCTIONS ===")
r249_fns = [
    (n, o) for n, o in r249_items
    if inspect.isfunction(o) and re.search(r"(22[8-9]|23[0-9]|24[0-5])$", n)
]
for r249_n, r249_o in r249_fns[-24:]:
    print(r249_n, r249_o.__code__.co_varnames[:r249_o.__code__.co_argcount],
          "references", r249_o.__code__.co_names[:24])

print("\n=== TARGET OBJECTS ===")
for r249_n, r249_o in r249_items:
    if re.search(r"truth|target|impulse|irf", r249_n, re.I) and not r249_n.startswith("r"):
        if isinstance(r249_o, np.ndarray):
            print(r249_n, r249_o.shape, r249_o if r249_o.size <= 24 else "")
        elif isinstance(r249_o, torch.Tensor):
            print(r249_n, tuple(r249_o.shape))
        elif isinstance(r249_o, dict):
            print(r249_n, "dict", list(r249_o.keys())[:10])
        elif isinstance(r249_o, (str, int, float)):
            print(r249_n, repr(r249_o)[:100])

with open("/work/notes.md", "a") as r249_file:
    r249_file.write(
        "\n## Runs 247-249: metadata recovery\n"
        "- Runs 247 and 248 stopped on inventory type errors, before any weight changes.\n"
        "- Using explicit Tensor/ndarray branches to recover the candidate and replay functions.\n"
        "- Python runs: 249; forward passes: 17.\n"
    )
```
