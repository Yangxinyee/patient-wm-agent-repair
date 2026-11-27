The recovery output was truncated before the safety results, so it adds no evidence for keeping the edit. I’ll now print a compact summary of factual fit, the saved dose projections, and the record-based effect conclusions. This should tell us whether to retain, adjust, or reject the current candidate.

```python
from pathlib import Path
import re, numpy as np, pandas as pd

def a292(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

checkpoint292 = {n: a292(load(n)).copy() for n in edited()}
print("CURRENT EDIT SAVED:", sorted(checkpoint292))

print("\nFORWARD32 FACTUAL GATE")
for p in sorted(Path("/work/fwd_32").glob("*")):
    if p.suffix == ".csv" and "factual" in p.name:
        d292 = pd.read_csv(p)
        print(p.name, "rows", len(d292), "columns", d292.columns.tolist())
        cs292 = [c for c in d292 if "nll" in c.lower()]
        print({c: float(pd.to_numeric(d292[c], errors="coerce").mean()) for c in cs292})
        ec292 = next((c for c in ("event", "event_name", "code") if c in d292), None)
        if ec292 and cs292:
            print(d292.groupby(ec292)[cs292].mean().to_string())
    elif p.suffix == ".json":
        print(p.name, p.read_text()[:1200])

print("\nEDIT OBJECTS")
for name292 in ("B286", "fits286", "safe236", "gains290", "P286", "ids", "tuning_ids", "tune_ids"):
    if name292 not in globals():
        continue
    obj292 = globals()[name292]
    if isinstance(obj292, dict):
        print(name292, "keys", list(obj292)[:30])
        if name292 == "fits286":
            for k292, v292 in list(obj292.items())[:3]:
                print(" sample", k292, type(v292).__name__,
                      tuple(v292.shape) if hasattr(v292, "shape") else str(v292)[:200])
    elif hasattr(obj292, "shape"):
        print(name292, type(obj292).__name__, tuple(obj292.shape))
    else:
        print(name292, str(obj292)[:500])

print("\nRELEVANT KERNEL NAMES")
selected292 = []
for n292, v292 in list(globals().items()):
    if not n292.startswith("_") and re.search(
        r"effect|truth|target|causal|beta|impulse|persist|safe|coef|tune.*id|ids.*tune", n292, re.I
    ):
        if isinstance(v292, dict):
            selected292.append(f"{n292}: dict keys={str(list(v292)[:10])[:160]}")
        elif hasattr(v292, "shape"):
            s292 = f"{n292}: {type(v292).__name__} {tuple(v292.shape)}"
            if isinstance(v292, np.ndarray) and v292.size <= 16:
                s292 += " " + str(v292)
            selected292.append(s292)
        elif isinstance(v292, (float, int, str, list, tuple)):
            selected292.append(f"{n292}: {str(v292)[:160]}")
print("\n".join(selected292)[:1700])

s292 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("\nSAFE236 AND MOST RECENT DECISIONS")
for part292 in s292:
    if re.match(r"## (?:Run |Runs )?(?:23[5-9]|24[0-2]|290|291)\b", part292):
        print(part292[:700])

print("\nEARLY DYNAMIC-EFFECT EVIDENCE")
remaining292 = 2700
for part292 in s292:
    m292 = re.match(r"## (?:Run |Runs )?(\d+)", part292)
    if m292 and int(m292.group(1)) <= 80 and re.search(
        r"dynamic|persist|impulse|true effect|causal|null|absent|identif|lagged", part292, re.I
    ):
        text292 = part292[:min(450, remaining292)]
        print(text292)
        remaining292 -= len(text292)
        if remaining292 <= 0:
            break

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 292\n"
            "Run291 output truncated before safety evidence. Saved current full edit checkpoint292 "
            "and performed compact recovery of Forward32 factual gate, projection objects, and "
            "record-based effect notes. No weights changed. Counts: 292 Python runs, 33 forwards.\n")
```
