The balanced `.25/.25` candidate raises the immediate norepinephrine response while only slightly weakening insulin, and its offline factual scores remain below 0.005. I’ll stage it for GPU validation. I’ll keep it only if factual fit holds and the full rollouts improve; otherwise I’ll restore the backed-up candidate.

```python
import torch
import numpy as np
import dis
from pathlib import Path

currentT275 = np.concatenate([
    load("attr_heads.value_bin.2.weight").numpy().T,
    load("attr_heads.value_bin.2.bias").numpy()[None, :]
], axis=0)
assert np.max(np.abs(currentT275-grid269[(.25,.15)]["T"])) < 2e-6

units275 = arch263["units"]
currentRows275 = np.concatenate([
    load("attr_heads.value_bin.0.weight").numpy()[units275],
    load("attr_heads.value_bin.0.bias").numpy()[units275, None]
], axis=1)
assert np.max(np.abs(currentRows275-arch263["rows"])) < 2e-5

T275 = grid269[(.25,.25)]["T"]
save_edit("attr_heads.value_bin.2.weight",
          torch.tensor(T275[:-1].T.copy(), dtype=torch.float32))
save_edit("attr_heads.value_bin.2.bias",
          torch.tensor(T275[-1].copy(), dtype=torch.float32))
print("STAGED .25/.25", grid269[(.25,.25)]["score"])

with Path("/work/notes.md").open("a") as f:
    f.write("""
## Run 275 / balanced broadened head staged
Verified current tensors against grid269(.25,.15) and arch263 rows.
Changed only final value-head weights/bias to grid269(.25,.25).
Offline deltas +.001181/+.004463/+.003732; immediate slopes
[-1.761,-1.829,+3.033,+1.981]. backup273 preserves the independently
validated .25/.15 candidate. Pending GPU factual and rollout tests.
""")

print("\n=== PAIRED-CONTRAST METADATA ===")
for key in [("insulin", .5), ("norepinephrine", .5)]:
    print(key, describe273(paired214[key]))
for n in ["targets179", "beta179", "family204"]:
    if n in globals():
        print(n, describe273(globals()[n]))

print("\n=== CONTRAST STATISTICS CONSTRUCTION ===")
print("constants", build_stats263.__code__.co_consts)
instructions275 = list(dis.get_instructions(build_stats263))
for ins in instructions275[-135:]:
    print(ins.offset, ins.opname, ins.argrepr)
```
