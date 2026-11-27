The measured baseline cost is **0.0028–0.0046 nats**, leaving little room for NE repair. I also found that stay-averaged rollout shifts differ from averages over all reached hours, so targets must use matching weights. Next I’ll check whether the rejected detector protected every state slot and recover the decoder needed to test factual corrections.

```python
import dis
import inspect
from pathlib import Path
import numpy as np
import torch

print("=== DETECTOR FIT ARRAYS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if name.endswith(tuple(str(k) for k in range(190, 200))) and isinstance(obj, (np.ndarray, torch.Tensor, dict)):
        if isinstance(obj, dict):
            print(name, "keys:", list(obj)[:8])
        else:
            print(name, tuple(obj.shape))

print("\n=== DETECTOR FUNCTION ===")
print("global names:", fit_detector197.__code__.co_names)
print("local names:", fit_detector197.__code__.co_varnames)
print(dis.Bytecode(fit_detector197).dis()[:4400])

print("\n=== DECODER FUNCTION CONTRACT ===")
print("global names:", decode_states175.__code__.co_names)
print("local names:", decode_states175.__code__.co_varnames)
print("constants:", [c for c in decode_states175.__code__.co_consts
                     if isinstance(c, (str, int, tuple))][-18:])

print("\n=== PAIRED TRAIN STATES AND DECODED OBJECTS ===")
p = Path("/work/fwd_22")
for f in sorted(p.glob("*.npz")):
    with np.load(f) as a:
        print(f.name, {k: a[k].shape for k in a.files})
for name in sorted(list(globals())):
    obj = globals()[name]
    if any(t in name.lower() for t in ["decoded", "meta175", "static175", "hidden182", "prob175", "panel", "causal"]):
        if isinstance(obj, dict):
            print(name, {k: getattr(v, "shape", type(v).__name__) for k, v in list(obj.items())[:8]})
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif hasattr(obj, "columns"):
            print(name, obj.shape, list(obj.columns)[:18])

print("\n=== INITIAL RECORD-BASED ESTIMATES ===")
ss = Path("/work/notes.md").read_text().split("\n## ")
for s in ss[:7]:
    print("\n## " + s[:650])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 203: investigate factual-detector failure mechanism\n"
            "Baseline factual cost ranges .002783 to .004636 on two samples. "
            "Stay-averaged rollout differs from reached-hour-pooled averages; targets "
            "must be compared using identical weighting. Inspecting detector slot coverage "
            "and paired-state decoder contracts before selecting a new repair. "
            "Python runs: 203; forwards: 23.\n")
```
