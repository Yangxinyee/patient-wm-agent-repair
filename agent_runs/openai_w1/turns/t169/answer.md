The last run failed before producing a usable result, so it does not validate the proposed repair. I’ll preserve and reset the outstanding edit, then recover the architecture and relevant estimates. The next step needs a reproducible drug-effect estimate and a weight pathway we can validate independently.

```python
import os, inspect, numpy as np, torch

recovery_numeric_154 = load("embed.numeric.weight").copy()
print("Outstanding edit distance:",
      float(np.linalg.norm(recovery_numeric_154 -
                           load_original("embed.numeric.weight"))))
drop_edit("embed.numeric.weight")
print("Current edits:", edited())

print("\n=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RELEVANT LIVE OBJECTS (bounded) ===")
tokens = ("causal", "effect", "beta", "posterior", "policy", "innovation",
          "iv_", "post_", "dose", "head")
items = [(k, v) for k, v in list(globals().items())
         if not k.startswith("_") and any(t in k.lower() for t in tokens)]
for k, v in sorted(items)[:65]:
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", repr(list(v.keys())[:8])[:220])
    elif callable(v):
        try:
            print(k, "function", inspect.signature(v))
        except Exception:
            print(k, "callable")
    elif isinstance(v, (int, float, str, list, tuple)):
        print(k, type(v).__name__, repr(v)[:180])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 155\n"
            "Run 154 indexing failure yielded no validated result. Preserved "
            "the outstanding numeric embedding in recovery_numeric_154 and "
            "restored the original before further diagnosis. Reading the "
            "architecture and bounded causal/posterior helper inventory. "
            "Counts: 155 Python runs, 8 forward passes.\n")
```