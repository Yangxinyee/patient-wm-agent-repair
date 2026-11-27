The recovery code stopped on a Tensor-versus-NumPy mismatch; it changed no weights. I’ll use tensor-safe copying and retry. Successful recovery should show the original weights restored and expose the relevant architecture and cached estimates.

```python
import os, inspect, numpy as np, torch

recovery_numeric_154 = torch.as_tensor(load("embed.numeric.weight")).detach().clone()
original_numeric_recovery = torch.as_tensor(load_original("embed.numeric.weight"))
print("Outstanding edit distance:",
      float(torch.linalg.vector_norm(recovery_numeric_154 -
                                     original_numeric_recovery)))
drop_edit("embed.numeric.weight")
print("Current edits:", edited())

print("\n=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RELEVANT LIVE OBJECTS (bounded) ===")
tokens_recovery = ("causal", "effect", "beta", "posterior", "policy",
                   "innovation", "iv_", "post_", "dose", "head")
items_recovery = [(k, v) for k, v in list(globals().items())
                  if not k.startswith("_")
                  and any(t in k.lower() for t in tokens_recovery)]
for k, v in sorted(items_recovery)[:65]:
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", repr(list(v.keys())[:8])[:200])
    elif callable(v):
        try:
            print(k, "function", inspect.signature(v))
        except Exception:
            print(k, "callable")
    elif isinstance(v, (int, float, str, list, tuple)):
        print(k, type(v).__name__, repr(v)[:160])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 155–156\n"
            "Run 155 failed on Tensor.copy before changing any weights. "
            "Run 156 uses tensor-safe cloning, preserves the outstanding "
            "numeric embedding in recovery_numeric_154, and restores the "
            "original. Recovering architecture and bounded live inventory. "
            "Counts: 156 Python runs, 8 forward passes.\n")
```