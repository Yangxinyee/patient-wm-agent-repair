The reduced edit has factual losses of 0.004325 and 0.002060 nats on two cohorts—a pooled increase of about 0.00319. That is encouraging, though rollout accuracy remains untested at this strength. I’ll compare earlier rollout results and calibration attempts before choosing the next test. An alternative is worthwhile only if it improves dose responses without spending too much factual-fit margin.

```python
import re
from pathlib import Path
import numpy as np
import pandas as pd

print("ROLLOUT AUDIT: equal-stay means on reached hours")
for k in range(1, 38):
    vals = {}
    for m in ["original", "edited"]:
        p = Path(f"/work/fwd_{k}/dose_edit_{m}.csv")
        if p.exists():
            d = pd.read_csv(p)
            mask = d["reached"].astype(str).str.lower().isin(["true", "1"])
            means = d.loc[mask].groupby(["edit", "stay_id"])["shift"].mean()
            vals[m] = {str(e): round(float(v), 4)
                       for e, v in means.groupby(level=0).mean().items()}
    if vals:
        print(k, vals)

print("\nSAVED INSULIN DECOMPOSITIONS")
arrays277 = {}
for fn in ["insulin_candidate58.npz", "insulin_candidate63.npz",
           "insulin_trial219.npz", "joint236.npz", "joint268.npz",
           "joint274.npz", "ne_candidate151.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as ar:
        arrays277[fn] = {k: ar[k].copy() for k in ar.files}
    print(fn, {k: (v.item() if v.size == 1 else tuple(v.shape))
               for k, v in arrays277[fn].items()})
for name in ["joint236.npz", "joint268.npz", "joint274.npz"]:
    ai = arrays277[name]["insulin_delta"].astype(float)
    for other in ["insulin_candidate58.npz", "insulin_candidate63.npz",
                  "insulin_trial219.npz"]:
        ar = arrays277[other]
        candidates = [(k, v) for k, v in ar.items()
                      if v.shape == (192, 192)]
        for k, v in candidates:
            av = v.astype(float)
            scale = float(np.sum(ai * av) / np.sum(av * av))
            relative_error = float(np.linalg.norm(ai - scale * av) /
                                   np.linalg.norm(ai))
            print(name, other, k, "best_scale", round(scale, 5),
                  "relative_error", round(relative_error, 5))

print("\nSELECTED NOTEBOOK EVIDENCE")
text277 = Path("/work/notes.md").read_text()
parts277 = re.split(r"(?=^## )", text277, flags=re.M)
chosen277 = []
for part in parts277:
    header = part.split("\n", 1)[0]
    relevant = (re.search(r"\b(?:Run|Runs)\s+(?:58|63|76|77|219|235|236)\b", header)
                or any(s in part.lower()
                       for s in ["temperature calibration", "aligned temperature",
                                 "calibration improved", "compensation failed"]))
    if relevant:
        chosen277.append(part[:1250])
print("\n".join(chosen277)[-6500:])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 277: paired evidence and candidate frontier\n"
            "Joint274 factual deltas: cohort100..199=.0043248285, "
            "cohort300..399=.0020602652; pooled about .00319. "
            "Recovering equal-stay rollout means, candidate decompositions and "
            "past compensation evidence before spending remaining validation passes. "
            "No weights changed. Python runs: 277; forwards: 37.\n")
```