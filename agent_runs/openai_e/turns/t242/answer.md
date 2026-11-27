The recovered results show that the insulin repair produces roughly −12.7/+4.3 glucose shifts and narrowly passes factual fit. The records support a positive norepinephrine effect, but stronger NE edits have exceeded the NLL allowance. I’ll inspect the exact target estimates and measured trade-offs; a smaller joint edit is useful only if its NE improvement survives factual validation.

```python
import json, re, numpy as np, pandas as pd
from pathlib import Path

secs210 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("=== EXACT BASELINE AND TARGET EVIDENCE ===")
for s in secs210:
    h = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"Run[s ]*(?:11|12|13|14|15|16|17|18|19|20|75|76|77|78|79|80|107|148|149|150|151|152|153|154|155|156|157|158)\b", h):
        print(s[:650])
print("\n=== FORWARD OUTPUT SCHEMAS ===")
for n in (13, 16, 19, 21, 23):
    p = Path(f"/work/fwd_{n}")
    print(n, [x.name for x in p.iterdir()])
    for x in sorted(p.glob("*.csv")):
        d = pd.read_csv(x)
        print(x.name, d.shape, d.columns.tolist())
        if n == 23:
            print(d.head(2).to_dict("records"))
print("\n=== ARCHIVED NE CANDIDATE SCHEMAS ===")
for filename in ("ne_reduced107.npz", "ne_candidate148.npz", "ne_candidate151.npz",
                 "ne_candidate154.npz", "ne_candidate155.npz", "ne_candidate199.npz"):
    with np.load("/work/" + filename) as z:
        print(filename, {k: z[k].shape for k in z.files})
print("\n=== HELPERS (NAMES ONLY) ===")
print([k for k,v in list(globals().items())
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\n=== CAUSAL OBJECTS (NAMES ONLY) ===")
print([k for k in list(globals())
       if re.search(r"(target|impulse|persist|causal|bootstrap|rd_|effect|panel|dose)", k, re.I)][:100])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 210: bounded target and measured-frontier recovery\n"
            "Insulin baseline passes narrowly and repairs the main response; NE evidence supports "
            "a positive effect but prior stronger edits fail factual fit. Recovering exact targets, "
            "construction archives, and GPU output schemas. No weights changed. "
            "Python runs: 210; forwards: 23.\n")
```
