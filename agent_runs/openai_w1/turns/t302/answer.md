The candidate clearly failed factual fit: loss rose from 1.532 to 5.829 nats. I’ll remove it immediately and inspect whether the damage grows along trajectories. If it does, that would point to recurrent feedback amplification and show why the frozen-state checks were insufficient.

```python
import numpy as np
import pandas as pd
import torch
import dis

rejected265 = arr262(load("core.b_in.weight")).copy()
drop_edit("core.b_in.weight")
print("Restored edits:", edited())

delta265 = arr262(_delta257).reshape(-1)
print("\n=== FEEDBACK DIAGNOSTICS ===")
print("delta_in scale quantiles:", np.quantile(delta265, [0, .1, .5, .9, 1]))
print("P.T @ B:", P264.T @ B257)
print("P.T @ (delta_in * B):",
      P264.T @ (delta265[:, None] * B257))
print("Added-weight top singular values:",
      np.linalg.svd(delta_b264, compute_uv=False)[:4])

fo265 = pd.read_csv("/work/fwd_28/factual_original.csv")
fe265 = pd.read_csv("/work/fwd_28/factual_edited.csv")
print("\n=== FACTUAL FILE SCHEMA ===")
print(fo265.columns.tolist())
print(fo265.head(2).to_string(index=False))
if len(fo265) == len(fe265):
    nllcols = [c for c in fo265 if "nll" in c.lower()]
    hourcols = [c for c in fo265 if "hour" in c.lower()]
    print("NLL columns:", nllcols, "hour columns:", hourcols)
    if nllcols and hourcols:
        ndiff265 = fe265[nllcols[0]].to_numpy() - fo265[nllcols[0]].to_numpy()
        hours265 = fo265[hourcols[0]].to_numpy()
        for lo, hi in ((0, 2), (2, 6), (6, 12), (12, 18), (18, 30)):
            m = (hours265 >= lo) & (hours265 < hi)
            print("hours", lo, hi, "n", m.sum(),
                  "mean NLL increase", np.mean(ndiff265[m]) if m.any() else None)

print("\n=== AVAILABLE TOKEN/STATE ARRAYS ===")
for name in sorted(list(globals())):
    v = globals()[name]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        s = tuple(v.shape)
        if len(s) == 3 and s[0] == 5215 and s[-1] == 192:
            print(name, s)
for name in ("slots258", "weights258", "ns258"):
    v = globals().get(name)
    if isinstance(v, dict):
        print(name, "weight names", len(v), list(v)[:6])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(name, type(v).__name__, tuple(v.shape))

print("\n=== E BUILDER OPERATIONS ===")
ops265 = [(i.opname, i.argrepr) for i in dis.Bytecode(E258)
          if i.opname in ("LOAD_GLOBAL", "LOAD_CONST", "LOAD_METHOD", "STORE_FAST")]
print(ops265)
fn265 = globals().get("r244_attention_features")
if callable(fn265):
    print("attention function arguments:",
          fn265.__code__.co_varnames[:fn265.__code__.co_argcount])
    print("attention function globals:", fn265.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 265\n"
            "Forward28 direct b_in candidate catastrophically failed factual fit: "
            "5.8288 versus 1.53191 nats on 5059 readings. Removed core.b_in edit immediately, "
            "restoring safe236. Frozen-state leakage checks did not predict trajectory damage; "
            "investigating feedback/stability before any further direct injection. "
            "Counts: 265 Python runs, 28 forwards.\n")
```