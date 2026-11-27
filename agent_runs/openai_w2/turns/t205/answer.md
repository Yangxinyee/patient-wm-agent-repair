The independent cohort also passes: NLL increased by **0.00309**. I’ll pool the factual results and audit how the records imply accumulated drug effects. Further strengthening is justified only if those estimates—not just the model’s rollout—support it.

```python
import re
import pandas as pd
import numpy as np
from pathlib import Path

pairs177 = []
for k in (20, 22):
    o = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    e = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    keys = ["stay_id", "hour", "event", "observed"]
    p = o.merge(e, on=keys, suffixes=("_original", "_edited"), validate="one_to_one")
    assert len(p) == len(o) == len(e)
    p["delta_nll"] = p["nll_edited"] - p["nll_original"]
    pairs177.append(p)
pooled177 = pd.concat(pairs177, ignore_index=True)
mean177 = pooled177["delta_nll"].mean()
influence177 = pooled177.assign(
    influence=pooled177["delta_nll"] - mean177
).groupby("stay_id")["influence"].sum()
G177 = len(influence177)
se177 = np.sqrt(G177 / (G177 - 1) * np.sum(influence177**2)) / len(pooled177)
print("Pooled factual readings/stays:", len(pooled177), G177)
print("Paired NLL increase and stay-cluster SE:", mean177, se177)
print(pooled177.groupby("event")["delta_nll"].agg(["mean", "count"]))

notes177 = Path("/work/notes.md").read_text()
sections177 = re.split(r"\n(?=## )", notes177)
print("\nNOTEBOOK HEADINGS")
heading_text177 = "\n".join(s.splitlines()[0] for s in sections177[1:])
print(heading_text177[:4300])

print("\nPERSISTENCE / IDENTIFICATION SUMMARIES")
for s in sections177:
    header = s.splitlines()[0].lower()
    if any(word in header for word in ("persistence", "latent", "identification",
                                       "state-model", "gaussian", "runs 157")):
        print(s[:550])

print("\nRELEVANT KERNEL VARIABLES")
matches177 = [n for n in sorted(list(globals()))
              if any(w in n.lower() for w in ("kalman", "ssm", "persist", "panel", "rho"))
              and not n.startswith("_")]
for n in matches177[:35]:
    x = globals()[n]
    if isinstance(x, dict):
        print(n, "dict", list(x.keys())[:10])
    elif isinstance(x, pd.DataFrame):
        print(n, "DataFrame", x.shape, list(x.columns)[:15])
    elif isinstance(x, np.ndarray):
        print(n, "array", x.shape)
    elif callable(x):
        print(n, "function")
    else:
        print(n, repr(x)[:150])

with open("/work/notes.md", "a") as f:
    f.write(f"\n## Forward22 / run177: independent and pooled validation\n"
            f"Current candidate factual increase +0.00309 on the independent "
            f"100-stay cohort. Across {G177} stays/{len(pooled177)} readings, "
            f"paired increase {mean177:.6f}, stay-cluster SE {se177:.6f}. "
            "Weights unchanged. Auditing record-based persistence and "
            "accumulated-effect identification before any further strengthening.\n")
```