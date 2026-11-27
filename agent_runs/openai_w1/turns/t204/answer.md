The gain-200 edit left factual NLL essentially unchanged, but also left the dose responses near baseline. That confirms it is safe but ineffective. I’ll recover the earlier experiments and available CPU helpers so I can test a more direct pathway without repeating failed approaches.

```python
import re, inspect
from pathlib import Path

nt182 = Path("/work/notes.md").read_text()
print("NOTEBOOK LENGTH", len(nt182))
print("\nEARLY NOTEBOOK")
print(nt182[:12000])

print("\nCORE TENSORS")
print("\n".join(f"{n}: {SHAPES[n]}" for n in NAMES
                if n.startswith("core.") and "prelude" not in n))

print("\nCALLABLE HELPERS")
helpers182 = []
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        helpers182.append(k + sig)
print("\n".join(helpers182))

print("\nTARGETED CACHE NAMES")
cache182 = []
for k, v in list(globals().items()):
    if k.startswith(("rp_", "r168_", "rr_input", "rr_state", "ds_")):
        shape = str(tuple(v.shape)) if hasattr(v, "shape") else type(v).__name__
        cache182.append(f"{k}: {shape}")
print("; ".join(cache182))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 182\nForward 15 confirms posterior gain-200 factual NLL delta "
            "+0.000000663 on 5,128 readings, but forward 16 confirms ineffective rollouts. "
            "Recovering prior experiments and CPU helpers to design a direct state/readout correction.\n")
```