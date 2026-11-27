The latest candidate preserved local factual fit, but insulin remained too weak and norepinephrine also moved glucose substantially. That does not yet validate a repair. I’ll recover the architecture and candidate construction before changing weights; a useful correction must improve the intended drug response without introducing unsupported cross-effects.

```python
import os, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

notes91 = open("/work/notes.md").read()
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(notes91[:12000])

def describe91(x, depth=0):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        if depth >= 2:
            return f"dict[{len(x)}]"
        return {str(k): describe91(v, depth + 1)
                for k, v in list(x.items())[:15]}
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}] " + str(
            [describe91(v, depth + 1) for v in x[:5]])
    if callable(x):
        try:
            return "callable " + str(inspect.signature(x))
        except Exception:
            return "callable"
    return repr(x)[:250]

print("\n=== RECENT CANDIDATE INTERFACES ===")
names91 = sorted(k for k in globals()
                 if not k.startswith("_") and k.endswith(("88", "89", "90")))
for k in names91:
    print(k, describe91(globals()[k]))

print("\n=== CORE/FEATURE HELPERS ===")
for k in sorted(list(globals())):
    x = globals()[k]
    if callable(x) and any(s in k.lower()
                          for s in ("core", "prior", "token", "head", "carrier")):
        print(k, describe91(x))

print("\nSubmitted edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 91: restart and token-route audit\n"
            "- Run 90 preserved local factual fit, but insulin response remained weak "
            "and NE produced substantial glucose cross-effects.\n"
            "- Recovering architecture, identification evidence, and candidate interfaces "
            "before choosing a drug-specific correction. No weights edited.\n"
            "- Python runs: 91; forward passes: 4.\n")
```