The asymmetric insulin candidate passed CPU screening at +0.00395 NLL; combining it with norepinephrine gain 1.4 left too little safety margin. I’ll save the insulin-only candidate for rollout validation. I’ll also recover the action-embedding helpers to check whether the norepinephrine gate recognizes the drug reliably across doubled and halved doses.

```python
import inspect
import dis
import numpy as np
import torch
from pathlib import Path

backup280 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
candidate280 = candidates279[(2.25, .75, 26.0, None)]
changes280 = {}
for n, v in candidate280.items():
    t = torch.as_tensor(v, dtype=torch.float32)
    old = torch.as_tensor(load(n), dtype=torch.float32)
    if not torch.equal(t, old):
        changes280[n] = float(torch.linalg.vector_norm(t - old))
        save_edit(n, t.clone())
print("SAVED ASYMMETRIC INSULIN", changes280)

for name in ("atom_exact65", "atom_clean60", "singleton_action_repair",
             "rp_action_encoder"):
    fn = globals()[name]
    print("\nHELPER", name, inspect.signature(fn))
    print("dependencies", fn.__code__.co_names)
    print("constants",
          [v for v in fn.__code__.co_consts
           if isinstance(v, (str, int, float, tuple))][:25])
print("\nEXACT ATOM WRAPPER")
dis.dis(atom_exact65)

print("\nGATE MODELS")
for event, obj in gate_models88.items():
    print(event, type(obj).__name__)
    if hasattr(obj, "coef_"):
        print("coef shape", obj.coef_.shape,
              "coef norm", float(np.linalg.norm(obj.coef_)),
              "intercept", obj.intercept_)
    elif isinstance(obj, dict):
        print("keys", list(obj))

print("\nEARLIER ACTION FIXTURE VARIABLES")
for k, v in list(globals().items()):
    if not k.startswith("_") and (
        any(s in k for s in ("183", "185"))
        or "grid" in k.lower() or "fixture" in k.lower()
    ):
        shape = getattr(v, "shape", None)
        if isinstance(shape, tuple):
            print(k, type(v).__name__, shape)
        elif isinstance(v, dict):
            print(k, "dict", len(v), list(v)[:5])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 280\n"
            "Provisionally saved asymmetric insulin pair, increase gain 2.25 / "
            "decrease gain .75 / gate 26, retaining pre-275 NE construction. "
            "CPU dNLL +.00395072; GPU validation pending. Backup backup280. "
            "Recovering action fixture helpers to test robust NE identity gating.\n")
```