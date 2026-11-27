The query-only pilot barely changed dose responses and badly damaged factual fit, so I’m rejecting it. Earlier, doubling recurrent input strengthened insulin responses but exceeded the NLL limit. I’ll now try a smaller gain of 1.35. Stronger insulin responses with acceptable factual fit would support this compromise; failure on either measure would rule it out.

```python
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Remove the entire failed compensated-query edit.
replay_ops109(baseline_ops109)
assert len(ops()) == 1
assert ops()[0]["name"] == "embed.numeric.weight"

# Previously, gain 2 improved rollouts but cost +0.02422 NLL.
# Test a smaller, reversible change.
recurrent_factor110 = 1.35
scale("core.b_in.weight", recurrent_factor110)
recurrent_candidate110 = snapshot_ops109()

print("Current operations:", ops())
print("Saved recurrent candidate:", len(recurrent_candidate110), "operations")

print("\n=== EARLIER RECURRENT DIAGNOSTIC FILES ===")
for k in (9, 10, 37, 38):
    folder = Path(f"/work/fwd_{k}")
    files = sorted(folder.glob("*.csv"))
    print("Forward", k, [p.name for p in files])
    for p in files:
        if p.name.startswith("factual"):
            df110 = pd.read_csv(p)
            print(p.name, "rows", len(df110),
                  "columns", list(df110.columns))
            nll_cols110 = [c for c in df110.columns if "nll" in c.lower()]
            for c in nll_cols110:
                if pd.api.types.is_numeric_dtype(df110[c]):
                    print(c, float(df110[c].mean()))

Path("/work/recurrent_135_110.json").write_text(
    json.dumps(json_ops109(recurrent_candidate110)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 110 / forwards 41-42: query pilot rejected\n"
            "- Compensated query strengths 0.5/0.5: factual NLL 1.70475 vs 1.63461,\n"
            "  increase +0.07014, dominated by MAP degradation.\n"
            "- Rollouts: insulin -2.427/+1.551; NE +0.112/+0.119.\n"
            "- No useful causal improvement; removed all 176 query/basis operations.\n"
            "- Restored numeric gain 4 and installed core.b_in.weight gain 1.35.\n"
            "- Earlier recurrent gain 2 gave insulin -3.768/+2.902, NE +0.259/-0.131,\n"
            "  but factual +0.02422. Testing a smaller gain rather than treating that as safe.\n"
            "- Current two-operation candidate is unvalidated. Forward passes: 42; 8 remain.\n")
```