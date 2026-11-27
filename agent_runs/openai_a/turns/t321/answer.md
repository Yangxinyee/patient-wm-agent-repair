The architecture confirms that the edit changes reading distributions, not the recurrent dynamics. The helper listing was truncated, so I’ll retrieve only the small pieces needed to reconstruct the trial and verify that the validated checkpoint is still installed.

```python
import inspect, re, numpy as np

print("FUNCTION NAMES:")
print([n for n, v in list(globals().items()) if inspect.isfunction(v)])

print("\nLAST-SCREEN AND ACTIVITY OBJECTS:")
for n, v in sorted(list(globals().items())):
    if ("274" in n or re.search(r"activity|^ag_|^act_|^gated", n, re.I)):
        if inspect.isfunction(v):
            print(n, inspect.signature(v), "dependencies:", v.__code__.co_names)
        elif isinstance(v, dict):
            print(n, "dict keys:", list(v)[:8])
        elif hasattr(v, "shape"):
            print(n, "shape:", tuple(v.shape))
        elif isinstance(v, (str, int, float, tuple, list)):
            print(n, repr(v)[:160])

print("\nGRAFT METADATA:")
for n in ("sc_names", "sc_nodes", "gate_rows274", "sc_gate_b",
          "selective_sep_b", "blend_nodes268", "blend_oldnodes268"):
    if n in globals():
        print(n, repr(globals()[n])[:300])

print("\nFALLBACK EQUALITY:")
for n in edited():
    print(n, float(np.max(np.abs(np.asarray(load(n)) -
                                np.asarray(validated_gated8_checkpoint[n])))))
```
