The replay now matches the saved outputs to about \(1.5\times10^{-5}\) RMS. It needs the current bundle in memory and the learned age embedding in seconds before projection. This supports using local finite differences. I’ll next rebuild a batch-independent readout and verify the dose carriers; correct dose contrasts and stable factual outputs would justify a transport pilot.

```python
import dis, inspect
import numpy as np
import pandas as pd
import torch

print("=== READOUT HELPER ===")
for ins in dis.get_instructions(means105):
    if ins.opname not in ("RESUME", "CACHE", "PRECALL", "PUSH_NULL",
                          "COPY", "SWAP"):
        print(ins.opname, ins.argrepr)

print("\n=== DATAFRAMES ===")
for n, x in list(globals().items()):
    if isinstance(x, pd.DataFrame):
        print(n, x.shape, list(x.columns)[:16])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
print("codes:", VOCAB["codes"])
print("config:", VOCAB["config"])
for n, x in VOCAB.items():
    if any(s in n.lower() for s in ("norm", "unit", "status")):
        print(n, repr(x)[:1600])

print("\n=== CANDIDATE COORDINATES ===")
print("154 variable names:",
      ", ".join(sorted(n for n in globals() if n.endswith("154"))))
for n in ("coords154", "slots154", "selected154", "plus154", "minus154",
          "cost154", "policy154"):
    if n in globals():
        x = globals()[n]
        if n == "policy154":
            print(n, [(float(np.linalg.norm(c)), float(b)) for c, b in x])
        else:
            print(n, repr(x)[:700])

origpost163 = {n: load_original(n) for n in post154}
zo163 = posterior153(
    torch.as_tensor(C154, dtype=torch.float32),
    torch.as_tensor(E154, dtype=torch.float32),
    torch.as_tensor(U154, dtype=torch.float32), origpost163)
zc163 = posterior153(
    torch.as_tensor(C154, dtype=torch.float32),
    torch.as_tensor(E154, dtype=torch.float32),
    torch.as_tensor(U154, dtype=torch.float32), post154)
print("Posterior shape:", tuple(zo163.shape))
dz163 = (zc163-zo163).numpy()
print("Largest per-coordinate posterior changes:",
      sorted([(float(np.std(dz163[:, j])),
               float(np.mean(dz163[:, j])), j)
              for j in range(dz163.shape[1])], reverse=True)[:8])

print("\n=== REFERENCE SUMMARY ===")
print(reference.columns.tolist())
print(reference.select_dtypes("number").mean().to_dict())
```
