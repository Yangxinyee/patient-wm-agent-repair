The last test changed the hidden slots substantially but barely changed the prediction output. The existing latent route therefore looks too weak for the repair. I’ll recover the architecture and retained helpers to locate the bottleneck; a useful route must affect dose responses while preserving factual fit.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== RETAINED HELPERS AND ARRAYS ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if callable(v) and any(s in k.lower() for s in
                          ("63", "64", "dose", "transition", "roll", "kernel", "causal", "effect", "extract")):
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k, type(v).__name__)
    elif hasattr(v, "shape") and any(s in k.lower() for s in
                                    ("63", "64", "dose", "kernel", "beta", "resid", "policy", "102", "104", "110")):
        print(k, type(v).__name__, v.shape)
notes112 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes112[:18000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 112 / selective-route recovery\n"
            "Large latent perturbations change slots but barely change predictive output; "
            "recovering the architecture and validated helpers before choosing a dose-residual route. "
            "No weights edited.\n")
```