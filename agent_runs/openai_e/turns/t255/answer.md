The rollout shows a much stronger insulin response than the original model, but norepinephrine still has little effect. I’ll recover the factual result and edit details before accepting this trial. A factual NLL increase below 0.005 would keep it viable; a larger increase would require reverting it.

```python
import os, json, re, numpy as np, pandas as pd
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== FORWARD 24 FILES ===")
for p in sorted(Path("/work/fwd_24").glob("*")):
    print(p.name, p.stat().st_size)
    if p.suffix in (".json", ".txt"):
        print(p.read_text()[:5000])
    elif p.suffix == ".csv":
        t = pd.read_csv(p)
        print("shape", t.shape, "columns", list(t.columns))
        print(t.head(2).to_string(index=False))
        if "nll" in t.columns:
            print("mean nll", t["nll"].mean())

print("\n=== SAVED EDIT SCHEMAS ===")
for fn in ("baseline77.npz", "insulin_candidate63.npz", "insulin_trial219.npz",
           "ne_candidate82.npz", "ne_candidate155.npz"):
    a = np.load("/work/" + fn)
    print(fn, {k: a[k].shape for k in a.files})

print("\n=== SELECTED PERSISTENT OBJECTS ===")
names = [k for k in list(globals()) if re.search(
    r"^(baseline|base|delta|direction|.*(?:77|81|82|162|175|199|214|219)|.*(?:decode|detector|target|causal|impulse|builder).*)$", k
)]
for k in names[:100]:
    v = globals()[k]
    if callable(v):
        print(k, "callable")
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(k, "dict keys", list(v)[:14])
    elif isinstance(v, (int, float, str)):
        print(k, repr(v)[:240])
    else:
        print(k, type(v).__name__)

notes = Path("/work/notes.md").read_text()
for term in ("Run 77", "Run 82", "Run 155", "Run 214", "Run 219"):
    hits = list(re.finditer(re.escape(term), notes))
    if hits:
        pos = hits[-1].start()
        print("\n=== NOTE:", term, "===\n", notes[max(0, pos-20):pos+1800])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 220: recover current trial validation\n"
            "Forward25 reports pooled reached-hour insulin shifts -12.474/+4.530 and NE -.203/-.794. "
            "These are predictions, not truth. Recovering Forward24 factual loss, architecture, "
            "and saved edit contracts before deciding whether trial219 is admissible. "
            "Python runs: 220; forwards: 25.\n")
```