The recovered targets are roughly −13.6/+6.8 for insulin and +4.2 to +4.6/−2.1 to −2.3 for norepinephrine. The insulin baseline is closer, but underestimates the halving response. I’ll check how it was constructed and recompute rollout summaries using only reached hours; averaging unreached hours would understate the effects. Agreement with the notebook summaries will confirm the comparison is valid.

```python
from pathlib import Path
import re, numpy as np, pandas as pd

notes180 = Path("/work/notes.md").read_text()
sections180 = re.split(r"(?=^## )", notes180, flags=re.M)
print("INSULIN BASELINE HISTORY")
for s180 in sections180:
    title180 = s180.split("\n", 1)[0]
    if (re.search(r"\b(?:63|64|65|66|67|68|69|70|71|72|73|74|75|76)\b", title180)
        and re.search(r"(insulin|baseline|causal|persistence|target)", s180, re.I)):
        print(s180[:700])

print("\nREACHED-HOUR ROLLOUT SUMMARIES")
for k180 in [1, 5, 7, 10, 12, 15, 18]:
    p180 = Path(f"/work/fwd_{k180}/dose_edit_edited.csv")
    if not p180.exists():
        p180 = Path(f"/work/fwd_{k180}/dose_edit_original.csv")
    d180 = pd.read_csv(p180)
    r180 = d180[d180["reached"].astype(str).str.lower().eq("true")]
    pooled180 = r180.groupby("edit")["shift"].mean()
    stay180 = r180.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").mean()
    print(k180, "stays", d180.stay_id.nunique(),
          "range", (d180.stay_id.min(), d180.stay_id.max()),
          "pooled", pooled180.round(4).to_dict(),
          "stay-weighted", stay180.round(4).to_dict())

print("\nVALIDATION IDs")
for k180 in ["VALID_IDS", "validation_ids", "validation_ids59",
             "validation_ids80", "ids107", "ids9_107"]:
    if k180 in globals():
        o180 = globals()[k180]
        print(k180, str(o180)[:650])

print("\nFEATURE / COUNTERFACTUAL OBJECTS")
keys180 = [k for k in list(globals())
           if re.search(r"(trial_direction|constrained_fits|normalization|cf_|prelude|action_|feature|readout|insulin)", k)]
for k180 in sorted(keys180):
    o180 = globals()[k180]
    if callable(o180):
        print(k180, "function")
    elif hasattr(o180, "shape"):
        print(k180, tuple(o180.shape))
    elif isinstance(o180, dict):
        print(k180, "dict", list(o180.keys())[:6])
    elif isinstance(o180, (list, tuple)):
        print(k180, "list", len(o180))
with Path("/work/notes.md").open("a") as f180:
    f180.write("\n## Run 180: reached-hour audit and insulin construction recovery\n"
               "Recovered target scales and baseline history; recomputed dose summaries on reached "
               "hours, including stay-weighted aggregation. No weights changed. "
               "Python runs: 180; forwards: 22.\n")
```
