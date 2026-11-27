The records support a multi-hour response: insulin lowers glucose, while norepinephrine raises MAP, after adjustment for pre-dose history. These estimates still depend on adequate confounding adjustment. I’ll inspect the available replay and value-head helpers next, looking for a localized edit that changes dose sensitivity without disturbing factual predictions.

```python
import types, inspect, pandas as pd

print("=== USER-DEFINED FUNCTIONS ===")
for k in sorted(list(globals())):
    v = globals()[k]
    if isinstance(v, types.FunctionType) and v.__module__ == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k, "?")

print("\n=== VALUE AND LATENT WEIGHTS ===")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "z_post", "z_prior", "z_in", "z_head", "last_numeric", "query_norm"]):
        print(n, SHAPES[n])

print("\n=== FRESH OBJECT INVENTORY (NO CONTENTS) ===")
for k in sorted(list(globals())):
    if k.startswith("rr_"):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys", list(v)[:8])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))

print("\n=== ORIGINAL ROLLOUT NOTES ===")
for s in rx_notes.split("\n## "):
    if ("dose_edit" in s or "rollout" in s.lower()) and len(s) < 2200:
        if any(t in s for t in ["Run 1:", "Run 2:", "Run 3:", "Run 4:", "Run 5:", "Run 6:", "Run 7:", "Run 8:", "Run 9:", "Run 10:", "Run 11:", "Run 12:", "Run 19:", "Run 20:", "Run 21:", "Run 22:"]):
            print("## " + s)

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 121\nRecovered six-hour adjusted/deconvolved response kernels: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690], NE [3.351,2.621,1.717,0.943,0.494,0.409]. These are record-based estimates under conditional exogeneity, not known simulator truth. Inspecting compact replay/head interfaces to construct a localized algebraic repair. No edits. Counts: 121 Python runs, 6 forward passes.\n")
```