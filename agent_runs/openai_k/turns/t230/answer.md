The retained objects include a zero-channel baseline and several earlier routing candidates, but their local diagnostics were not reliable. I’ll compare the official results from all previous tests before choosing another direction. Correct drug responses with acceptable factual loss would support reusing a candidate; otherwise, the routing needs a different calibration.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import torch

print("=== OFFICIAL TEST HISTORY ===")
for folder in sorted(Path("/work").glob("fwd_*"),
                     key=lambda p: int(p.name.split("_")[-1])):
    facts = {}
    for p in sorted(folder.glob("factual_*.csv")):
        t = pd.read_csv(p)
        facts[p.stem.removeprefix("factual_")] = float(t["nll"].mean())
    rolls = {}
    for p in sorted(folder.glob("dose_edit_*.csv")):
        t = pd.read_csv(p)
        valid = t["reached"].astype(bool) if "reached" in t else np.ones(len(t), bool)
        t = t.loc[valid]
        rolls[p.stem.removeprefix("dose_edit_")] = {
            str(k): round(float(v), 5)
            for k, v in t.groupby("edit")["shift"].mean().items()
        }
    print(folder.name, "NLL", facts, "rollout", rolls,
          "states", [p.name for p in folder.glob("*.npz")])

def np209(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("\n=== RESERVED-COLUMN NORMS ===")
for name in ("baseline138", "candidate139", "candidate193", "backup199", "best188"):
    obj = globals().get(name)
    if isinstance(obj, dict):
        if "z_in.weight" in obj:
            a = np209(obj["z_in.weight"])
            base = np209(baseline138["z_in.weight"])
            print(name, "cols30:32 norms", np.linalg.norm(a[:,30:32],axis=0).round(5),
                  "delta norms", np.linalg.norm(a[:,30:32]-base[:,30:32],axis=0).round(5))
        elif "dirs" in obj:
            a = np209(obj["dirs"])
            print(name, "dirs shape", a.shape,
                  "norms", np.linalg.norm(a,axis=0).round(5))

print("\n=== SAVED SOURCE STRINGS ===")
print([(k, len(v)) for k,v in list(globals().items())
       if isinstance(v,str) and ("def r132_channels" in v or
          "def dynamic_delta154" in v or "def repaired_core_local" in v
          or "def r186_heads" in v)])

print("\n=== EARLIER NOTE CONCLUSIONS ===")
lines209 = Path("/work/notes.md").read_text().splitlines()
hits209 = [line for line in lines209
           if any(s in line.lower() for s in
                  ("forward ", "factual delta", "dose response", "causal effect",
                   "impulse", "innovation", "identification"))]
for line in hits209[-28:]:
    print(line[:300])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 209: official test-history comparison\n"
            "- Comparing all official rollout/factual results before selecting new routing directions; local-head diagnostics alone were refuted.\n"
            "- No weights changed. Python runs: 209; forward passes: 14.\n")
```