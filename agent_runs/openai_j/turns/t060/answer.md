Doubling the bins weakened both insulin responses and did not consistently improve norepinephrine, so I’ll reject it. Next I’ll investigate an action-minus-policy correction in the latent pathway. A useful correction must carry dose changes while remaining nearly zero on factual records.

```python
import numpy as np
import pandas as pd
import torch
import dis

for name in ["embed.bins.dose.weight", "embed.bins.rate.weight"]:
    drop_edit(name)
print("Active edits after rejection:", edited())

print("\n=== VOCAB SUMMARY ===")
print("Keys:", list(VOCAB))
print("Codes:", VOCAB["codes"])
print("Config:", VOCAB["config"])

print("\n=== SMALL DATAFRAMES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and len(obj) < 12000:
        print(name, obj.shape, list(obj.columns))

print("\n=== STATE/FEATURE OBJECTS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and "x" in obj and "stay_id" in obj:
        print(name, "state dictionary", {k: getattr(v, "shape", None) for k, v in obj.items()})
    elif isinstance(obj, (np.ndarray, torch.Tensor)) and obj.ndim == 2:
        if obj.shape[1] in [32, 64, 192, 384, 576, 768, 992] and obj.shape[0] < 12000:
            print(name, type(obj).__name__, tuple(obj.shape))

print("\n=== HELPER IMPLEMENTATIONS ===")
for helper_name in ["wt", "mlp", "latent_context_at", "encode_single_debug"]:
    fun = globals()[helper_name]
    print("\n", helper_name)
    print("locals:", fun.__code__.co_varnames)
    print("names:", fun.__code__.co_names)
    print("constants:", tuple(x for x in fun.__code__.co_consts
                              if isinstance(x, (str, int, float, tuple))))
    if helper_name in ["wt", "mlp"]:
        dis.dis(fun)

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward passes 8–9; run 50: rejected bin gain\n"
            "- Bin gain factual NLL +0.00029, but insulin x2/half changed "
            "from -3.680/+2.018 to -3.198/+1.436; NE x2/half from "
            "+0.465/-0.871 to +0.552/-0.642. Restored both tensors.\n"
            "- Simple input gains have failed. Next investigate a "
            "history-conditioned action innovation in the posterior pathway.\n"
            "- Nine forward passes used. No active edits.\n")
```
