The 1,250-second trial predicts a larger insulin effect than the feasible 900-second version, but doubling and halving remain asymmetric. Before retaining it, I’ll recover its factual score and the record-derived effect estimates. It is only a better candidate if factual NLL stays within the limit and the stronger responses move toward those estimates.

```python
from pathlib import Path
import re
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_path = Path("/work/notes.md")
notes_text = notes_path.read_text()
print("\n=== EARLIER IDENTIFICATION AND TARGET NOTES ===")
sections = re.split(r"(?=^## )", notes_text, flags=re.M)
selected = [
    s for s in sections
    if re.search(
        r"target|causal|identif|instrument|randomiz|true effect|record-derived|"
        r"regression|confound|estimated effect|AR response",
        s, re.I
    )
]
remaining = 18000
for s in selected:
    if remaining <= 0:
        break
    excerpt = s[:min(len(s), 2200, remaining)]
    print(excerpt)
    remaining -= len(excerpt)

print("\n=== RECENT FORWARD FILES AND FACTUAL SCORES ===")
for dirname in ["fwd_30", "fwd_31", "fwd_32", "fwd_33", "fwd_34"]:
    directory = Path("/work") / dirname
    if not directory.exists():
        continue
    print(dirname, sorted(p.name for p in directory.iterdir()))
    for p in sorted(directory.glob("factual_*.csv")):
        df = pd.read_csv(p)
        print(p.name, "rows:", len(df), "columns:", df.columns.tolist())
        print(df.head(2).to_string(index=False))
        nll_cols = [c for c in df.columns if "nll" in c.lower()]
        for c in nll_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                print(c, "mean:", float(df[c].mean()))
                for event_col in ["event", "code", "event_code", "name"]:
                    if event_col in df.columns:
                        print(df.groupby(event_col)[c].agg(["mean", "count"]).to_string())
                        break

print("\n=== RELEVANT PERSISTENT VARIABLES ===")
snapshot = list(globals().items())
for k, v in sorted(snapshot):
    if k.startswith("_"):
        continue
    if re.search(
        r"BASE|center|target|effect|coef|tau|persist|checkpoint|tuning|stay|"
        r"fit|causal|dose|strength|trial|flow|restore|calib",
        k, re.I
    ):
        shape = getattr(v, "shape", None)
        if isinstance(v, dict):
            detail = "keys=" + repr(list(v.keys())[:18])
        elif isinstance(v, (list, tuple)):
            detail = "len=" + str(len(v)) + "; first=" + repr(v[:8])[:250]
        elif isinstance(v, (str, int, float, np.number)):
            detail = repr(v)[:250]
        elif callable(v):
            detail = "callable"
        else:
            detail = "shape=" + str(shape)
        print(k, type(v).__name__, detail)

with notes_path.open("a") as f:
    f.write(
        "\n## Run 287: restart recovery and acceptance audit\n"
        "- Tau=1250 rollout: insulin x2 -6.728, half +1.549; "
        "NE x2 +4.374, half -0.953 on tuning A.\n"
        "- Recovering its paired factual score and earlier record-derived targets "
        "before retaining or changing it. No weights changed.\n"
    )
```