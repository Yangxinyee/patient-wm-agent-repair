The architecture confirms that dose effects must survive feedback through generated readings, not just change the next prediction. The early record analyses also support a negative insulin effect, while norepinephrine estimates depend strongly on adjustment. I’ll recover the later identification results and rollout IDs; those should tell us which targets are credible enough to guide the next edit.

```python
import re
print("NOTEBOOK HEADINGS")
heads = re.findall(r"^##.*$", notes_text, flags=re.M)
print("\n".join(heads))
print("\nCAUSAL EVIDENCE EXCERPTS")
for term in ["regression discontinuity", "RD estimate", "RD evidence", "norepinephrine identification", "threshold", "placebo"]:
    hits = [m.start() for m in re.finditer(re.escape(term), notes_text, flags=re.I)]
    print("TERM", term, "matches", len(hits))
    for pos in hits[-2:]:
        print(notes_text[max(0,pos-180):pos+1100])
print("\nVARIABLE NAMES")
keys = [k for k in globals() if not k.startswith("_")]
print(" ".join(keys))
print("\nFORWARD FILES", [p.name for p in Path("/work/fwd_14").iterdir()])
print("\nTUNING STAY IDS")
import pandas as pd
print(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().head(100).tolist())
```
