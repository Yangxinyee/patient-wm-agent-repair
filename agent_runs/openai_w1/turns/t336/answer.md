The latest rollout strengthens insulin’s predicted glucose effect, but that does not yet establish correctness. I’ll recover the record-based effect estimates and the missing factual-fit result before changing weights. Agreement with those estimates and an NLL increase no greater than 0.005 would support keeping this candidate; either failure would argue against it.

```python
from pathlib import Path
import json, re, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

notes291 = Path("/work/notes.md").read_text()
sections291 = re.split(r"(?=^## )", notes291, flags=re.M)
print("=== EARLY IDENTIFICATION NOTES ===")
print(notes291[:3500])
hits291 = [
    s for s in sections291
    if re.search(r"identif|true effect|causal effect|record.effect|instrument|randomiz|innovation|safe236", s, re.I)
]
budget291 = 13500
for s in hits291:
    if budget291 <= 0:
        break
    text = s[:min(1600, budget291)]
    print(text)
    budget291 -= len(text)
print("=== RECENT NOTES ===")
print(notes291[-7000:])

print("=== FORWARD 32 FILES AND FACTUAL RESULTS ===")
for p in sorted(Path("/work/fwd_32").glob("*")):
    print(p.name, p.stat().st_size)
    if p.suffix == ".json":
        print(p.read_text()[:6000])
    elif p.suffix == ".csv" and "factual" in p.name:
        d = pd.read_csv(p)
        print("columns", d.columns.tolist(), "rows", len(d))
        print(d.head(2).to_string(index=False))
        for c in d.columns:
            if "nll" in c.lower():
                print(c, pd.to_numeric(d[c], errors="coerce").mean())
        eventcol = next((c for c in d.columns if c in ("event", "event_name", "code")), None)
        nllcols = [c for c in d.columns if "nll" in c.lower()]
        if eventcol and nllcols:
            print(d.groupby(eventcol)[nllcols].mean().to_string())

print("=== KERNEL RECOVERY ===")
snapshot291 = list(globals().items())
print("Names:", " ".join(sorted(
    n for n, v in snapshot291
    if not n.startswith("_") and n not in ("In", "Out", "get_ipython")
)))
for n, v in snapshot291:
    if re.search(r"(236|286|289|290)$|effect|target|causal|truth|train_ids|tuning_ids", n, re.I):
        if isinstance(v, dict):
            print(n, "dict keys", list(v)[:35])
        elif isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)) or hasattr(v, "shape"):
            print(n, type(v).__name__, "shape", tuple(v.shape))
            if isinstance(v, np.ndarray) and v.size <= 30:
                print(v)
        elif isinstance(v, (float, int, str, tuple, list)):
            print(n, str(v)[:900])
print("Currently edited:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 291\n"
            "Forward33 current [-3,.4] recurrent candidate predicts shifts "
            "[-4.013,+1.672,+.279,-.465], versus original [-1.731,+.798,+.048,-.481]. "
            "Recovering record-based effect estimates and Forward32 factual gate before further edits. "
            "No weights changed. Counts: 291 Python runs, 33 forwards.\n")
```
