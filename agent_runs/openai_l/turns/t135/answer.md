The last experiment found a nearly lossless dose-information channel and a controller that could preserve a private state. That is promising, but factual fit and complete rollouts remain untested. I’ll recover the causal estimates and helpers before assembling a reversible candidate. It must improve dose responses while staying within the 0.005-nat factual-loss limit.

```python
import os, re, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

notes126 = open("/work/notes.md").read()
print("\n=== EARLY NOTEBOOK: CAUSAL EVIDENCE AND TARGETS ===")
print(notes126[:18000])
print("\n=== NOTEBOOK SECTION INDEX ===")
print("\n".join(re.findall(r"^#{1,3} .+$", notes126, flags=re.M)))

g126 = dict(globals())
print("\n=== GLOBAL VARIABLE NAMES ===")
print(" ".join(sorted(k for k in g126 if not k.startswith("_"))))

print("\n=== HELPER FUNCTION SIGNATURES ===")
for k, v in sorted(g126.items()):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k, "(signature unavailable)")

print("\n=== RUN-125 COMPONENT DESCRIPTORS ===")
for k, v in sorted(g126.items()):
    if k.endswith("125"):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            print(k, type(v).__name__, a.shape, a.dtype,
                  "range", (float(np.nanmin(a)), float(np.nanmax(a))) if a.size else None)
        elif isinstance(v, dict):
            print(k, "dict", {str(x): getattr(y, "shape", type(y).__name__)
                             for x, y in list(v.items())[:25]})
        elif isinstance(v, (str, int, float, tuple, list)):
            print(k, repr(v)[:700])
        else:
            print(k, type(v).__name__)

print("\n=== TENSOR NAMES AND SHAPES ===")
for k in NAMES:
    print(k, SHAPES[k])
print("\nCurrent edits:", edited())
```
