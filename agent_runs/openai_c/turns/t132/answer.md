Uniform attention weakened insulin sensitivity, so the learned attention appears to carry useful dose information rather than suppress it. I’ll test a twofold query gain next. I’ll also inspect rate-bin coverage to see whether counterfactual doses reach poorly supported regions of the training records.

```python
from pathlib import Path
import json, numpy as np

while len(ops()) > 1:
    undo()
scale("core.prelude.0.attn.q.weight", 2.0)
print("CANDIDATE:", ops())

print("\nVOCABULARY BIN STRUCTURE")
for k94, v94 in VOCAB.items():
    if any(s94 in k94.lower() for s94 in ["rate", "dose", "normal"]):
        if isinstance(v94, dict):
            print(k94, list(v94.keys())[:8])
            for code94 in ["insulin", "norepinephrine", "8", "9"]:
                if code94 in v94:
                    print(code94, str(v94[code94])[:600])
        elif isinstance(v94, list):
            print(k94, "length", len(v94))
            print(str(v94)[:600])
        else:
            print(k94, str(v94)[:600])

print("\nDRUG RATE DISTRIBUTIONS")
print(train.columns.tolist())
for drug94 in ["insulin", "norepinephrine"]:
    if "event" in train.columns and "rate" in train.columns:
        rates94 = train.loc[train["event"].eq(drug94), "rate"].dropna()
        print(drug94, len(rates94),
              rates94.quantile([0, .1, .5, .9, .95, .99, 1]).to_dict())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 94 / forward 35: attention diagnostic\n"
            "- Uniform prelude attention weakened insulin responses to -0.188/+0.409; NE +0.060/-0.063.\n"
            "- Rejected and removed uniform-attention edit.\n"
            "- Installed numeric gain 4 plus prelude query weight x2 as a sharper-attention diagnostic.\n"
            "- Factual safety and rollout benefit unknown. Forward passes used: 35; 15 remain.\n")
```