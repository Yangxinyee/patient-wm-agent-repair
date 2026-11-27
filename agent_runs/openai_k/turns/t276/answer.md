The notes support persistent insulin effects and faster-decaying MAP effects, conditional on the dose-identification assumptions. The inventory failed because a NumPy function was mistaken for an array; no weights changed. I’ll fix that inspection and recover the candidate directly, then test whether its factual cost permits useful amplification.

```python
import torch, inspect, re, numpy as np

r248_items = list(globals().items())
print("=== CANDIDATE OBJECTS ===")
for r248_n, r248_o in r248_items:
    if not re.search(r"(243|244|245)$", r248_n):
        continue
    r248_t = type(r248_o).__name__
    if isinstance(r248_o, (torch.Tensor, np.ndarray)):
        print(r248_n, r248_t, tuple(r248_o.shape),
              np.asarray(r248_o.detach().cpu() if isinstance(r248_o, torch.Tensor) else r248_o).round(5)
              if r248_o.numel() <= 24 and isinstance(r248_o, torch.Tensor)
              else (r248_o.round(5) if isinstance(r248_o, np.ndarray) and r248_o.size <= 24 else ""))
    elif isinstance(r248_o, dict):
        print(r248_n, "dict", list(r248_o.keys())[:12])
    elif inspect.isfunction(r248_o):
        print(r248_n, "fn", r248_o.__code__.co_varnames[:r248_o.__code__.co_argcount])
    elif isinstance(r248_o, (list, tuple)):
        print(r248_n, r248_t, "length", len(r248_o))
    elif isinstance(r248_o, (str, float, int)):
        print(r248_n, r248_t, repr(r248_o)[:160])
    else:
        print(r248_n, r248_t)

print("\n=== RECENT REPLAY FUNCTIONS ===")
r248_fns = [
    (n, o) for n, o in r248_items
    if inspect.isfunction(o) and re.search(r"(22[8-9]|23[0-9]|24[0-5])$", n)
]
for r248_n, r248_o in r248_fns[-18:]:
    print(r248_n, r248_o.__code__.co_varnames[:r248_o.__code__.co_argcount])
    if re.search(r"core|head|finish|mean|corr|candidate", r248_n, re.I):
        try:
            print(inspect.getsource(r248_o)[:3500])
        except (OSError, TypeError):
            print("Source unavailable; referenced names:", r248_o.__code__.co_names)

print("\n=== RETAINED TARGET OBJECTS ===")
for r248_n, r248_o in r248_items:
    if not re.search(r"truth|target|impulse|irf", r248_n, re.I):
        continue
    if isinstance(r248_o, (torch.Tensor, np.ndarray)):
        print(r248_n, tuple(r248_o.shape))
        if isinstance(r248_o, np.ndarray) and r248_o.size <= 32:
            print(r248_o)
    elif isinstance(r248_o, dict):
        print(r248_n, "dict keys", list(r248_o)[:12])
    elif isinstance(r248_o, (str, int, float)):
        print(r248_n, repr(r248_o)[:180])

with open("/work/notes.md", "a") as r248_file:
    r248_file.write(
        "\n## Runs 247-248: recover candidate safely\n"
        "- Run 247 inventory failed by treating a NumPy module's shape function as an array; no weights changed.\n"
        "- Record impulses: glucose [-2.073,-2.235,-2.145,-1.439,-1.281,-0.905]; MAP [2.952,2.496,1.628,1.029,.343,.214].\n"
        "- These remain conditional on dose innovations being exogenous given recorded history.\n"
        "- Python runs: 248; forward passes: 17.\n"
    )
```
