The helpers are still available, but printing their tensor defaults again hid the details we need. I’ll keep this inspection compact and check the exact head tensors, numeric inputs, and latent scales. A convention error should explain the systematic bias, not merely reduce random prediction differences.

```python
import dis
import inspect
import numpy as np
import torch

print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in (
        "attr_heads.value", "last_", "query", "type_emb",
        "z_prior", "z_head", "gap", "flow", "out_norm", "embed.numeric"
    )):
        print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for key in VOCAB:
    if any(s in key.lower() for s in ("norm", "numeric")):
        print(key, str(VOCAB[key])[:1800])

print("\n=== METADATA ===")
print("head_meta columns:", list(head_meta.columns))
print(hm.head(4).to_string(index=False))
print("numeric rows:", fresh_head["numbers"][:4].numpy())
print("dt range:", float(dt.min()), float(dt.max()))
print("prior mean range:", float(mean_z.min()), float(mean_z.max()))
print("logsd range/quantiles:",
      np.quantile(logsd_z.numpy(), [0, .1, .5, .9, 1]))

print("\n=== HELPER IMPLEMENTATION DETAILS ===")
for name in ("wt", "ln", "mlp", "activation", "reconstruct_reading_head"):
    func = globals()[name]
    code = func.__code__
    constants = [
        v for v in code.co_consts
        if isinstance(v, (str, float, int, tuple)) and len(str(v)) < 300
    ]
    print(name, "names:", code.co_names, "constants:", constants)
    try:
        source = inspect.getsource(func)
        print(source[:2800])
    except (OSError, TypeError):
        if name in ("wt", "ln", "mlp", "activation"):
            instructions = list(dis.get_instructions(func))
            print([(i.opname, i.argrepr) for i in instructions][:55])

print("\n=== STATE ARCHIVE KEYS ===")
for key, obj in list(globals().items()):
    if isinstance(obj, np.lib.npyio.NpzFile):
        print(key, obj.files)
```
