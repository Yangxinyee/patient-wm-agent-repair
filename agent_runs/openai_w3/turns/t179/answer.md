The recurrent helper does not appear to apply the learned age embedding itself, although the architecture requires one. That could explain the replay error. I’ll test the age embedding and its insertion point against the saved states; a sharp error reduction would confirm this diagnosis, while unchanged error would point elsewhere.

```python
import dis
import numpy as np
import torch

print("=== ATTENTION AGE HANDLING ===")
for ins in dis.get_instructions(attention82):
    if ins.opname not in ("RESUME", "CACHE", "PRECALL", "PUSH_NULL",
                          "COPY", "SWAP"):
        print(ins.opname, ins.argrepr)

print("\n=== STATE INPUT VARIABLES ===")
for n, x in list(globals().items()):
    if isinstance(x, (np.ndarray, torch.Tensor)) and (
        n.lower().startswith(("tok", "hflow", "hprev", "h79", "ctx79", "prev"))
        or n in ("flowed", "E_original155", "positions155", "indices155")):
        print(n, tuple(x.shape))

print("\n=== CONTEXT HELPER RETURN ===")
cf161 = context_with_flow(indices155, 1.0)
if isinstance(cf161, tuple):
    print([tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
           for v in cf161])
else:
    print(type(cf161).__name__, tuple(cf161.shape))

print("\n=== ORIGINAL CACHE CHECK ===")
for n in ("core.c_out.weight", "core.b_in.weight",
          "core.blocks.0.0.cross.kv.weight", "core.age.0.weight"):
    if n in W:
        print(n, float(torch.max(torch.abs(
            W[n] - torch.as_tensor(load_original(n))))))

print("\n=== AGE-EMBEDDING TEST INPUTS ===")
m161, mask161, rawage161 = memory82(indices155, False)
for unit161 in (1.0, 3600.0):
    ae161 = mlp(torch.log1p(rawage161 * unit161), "core.age")
    print("age unit", unit161, "embedded shape", tuple(ae161.shape),
          "RMS", float(ae161.square().mean().sqrt()))
```
