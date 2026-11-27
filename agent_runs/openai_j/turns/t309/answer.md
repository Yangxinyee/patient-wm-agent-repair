This repeats the same rollout report, so it adds no new validation. I’ll inspect the earlier action-input probes and factual dose support before using another forward pass. A promising route must change dose sensitivity without merely exploiting the tested multipliers.

```python
import re
from pathlib import Path
import numpy as np
import pandas as pd

parts269 = ["DATA DOCUMENTATION:\n" +
            Path("/data/README.md").read_text()[:1600]]

rates269 = pd.read_csv("/data/train.csv", usecols=["event", "rate", "dose"])
parts269.append("\nTRAINING DOSE SUPPORT:")
for drug in ("insulin", "norepinephrine"):
    rr = rates269.loc[rates269["event"].eq(drug), ["rate", "dose"]]
    positive = rr.loc[rr["rate"].gt(0), "rate"]
    parts269.append(
        f"{drug}: rows={len(rr)}, positive={len(positive)}, "
        f"unique_positive_rates={positive.nunique()}\n"
        f"rate_quantiles={positive.quantile([0,.01,.1,.25,.5,.75,.9,.99,1]).to_dict()}\n"
        f"most_common_rates={positive.value_counts().head(10).to_dict()}")
    ratio = rr.loc[rr["rate"].gt(0), "dose"] / positive
    parts269.append("dose/rate quantiles=" +
                    str(ratio.quantile([0,.5,1]).to_dict()))

notes269 = Path("/work/notes.md").read_text()
sections269 = re.split(r"(?=^## )", notes269, flags=re.M)
action269 = [s for s in sections269 if re.search(
    r"numeric\.weight|action.input|input.probe|bin.compens|posterior.only|action.path",
    s, re.I)]
parts269.append("\nPREVIOUS ACTION-PATH EXPERIMENTS:")
seen269 = set()
for s in action269[:6] + action269[-3:]:
    if s in seen269:
        continue
    seen269.add(s)
    parts269.append(s if len(s) <= 600 else
                    s[:300] + "\n...\n" + s[-280:])

parts269.append("\nVOCAB KEYS:\n" + str(list(VOCAB.keys())))
out269 = "\n".join(parts269)
print(out269[:9000])
print("\nDiagnostic characters:", len(out269), "; current edits:", edited())

if "## Run 269: calibrated rollout and action-path audit" not in notes269:
    with open("/work/notes.md", "a") as f:
        f.write("\n## Run 269: calibrated rollout and action-path audit\n"
                "- Forward 33 / fwd_34: calibrated candidate factual dNLL +0.00541 on tuning 200–299; not yet pooled-validated.\n"
                "- Forward 34 / fwd_35: insulin -10.190/+5.661; NE +1.166/-1.765 on first 100 tuning stays.\n"
                "- NE improves over the fallback, but doubling remains below the assumption-dependent kernel estimate.\n"
                "- Reviewing earlier action-path probes and factual dose support; no weights changed.\n"
                "- 34 forward passes used.\n")
```