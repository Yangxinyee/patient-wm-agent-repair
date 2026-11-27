The original rollouts are much weaker than the provisional record-based targets, especially for doubled insulin. The token probe also confirms that the existing latent route has almost no effect on predicted values. I’ll inspect attention to action and latent tokens, along with weight caches; meaningful action attention would support a selective upstream repair, while negligible attention would require opening a new route.

```python
import dis
import numpy as np
import torch

print("=== CONFIG AND CAUSAL KERNELS ===")
print(VOCAB["config"])
for key114 in ("kernels19", "kernel22", "rich_fits", "r32_best"):
    obj114 = globals().get(key114)
    if obj114 is not None:
        out114 = repr(obj114)
        print(key114, out114[:1600])

print("\n=== ATTENTION IMPLEMENTATION ===")
for ins114 in dis.get_instructions(attention63):
    print(ins114.opname, ins114.argrepr)

print("\n=== TOKEN PROBE IMPLEMENTATION ===")
for ins114 in dis.get_instructions(token_probe110):
    print(ins114.opname, ins114.argrepr)

print("\n=== WEIGHT CACHES ===")
for key114, obj114 in list(globals().items()):
    if isinstance(obj114, dict):
        overlap114 = set(obj114).intersection(NAMES)
        if len(overlap114) > 5:
            print(key114, "cached weights:", len(overlap114))
print("\n=== TOKEN SCALES ===")
for j114, label114 in enumerate(("slot0", "slot1", "slot2", "slot3", "observation", "action", "latent")):
    t114 = tokens63[:, j114]
    print(label114, "RMS", float(t114.square().mean().sqrt()),
          "SD across dimensions", float(t114.std(dim=-1).mean()))
print("\n=== LAYER DEPENDENCIES ===")
print(layer63.__code__.co_names, layer63.__code__.co_consts)
print("Edits:", edited())
```