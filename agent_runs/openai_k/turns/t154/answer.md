The larger reversed channels strengthened insulin’s response only slightly and did not fix norepinephrine. I’ll check the factual-fit result and recover the architecture and calibration code before changing direction. A useful repair must improve both drug responses without exceeding the NLL limit.

```python
from pathlib import Path
import json, inspect, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

print("\n=== FORWARD 8 FILES ===")
for p in sorted(Path("/work/fwd_8").iterdir()):
    print(p.name, p.stat().st_size)
    if p.suffix == ".json":
        print(p.read_text()[:12000])
    elif p.suffix == ".csv":
        x = pd.read_csv(p)
        print(x.shape, x.columns.tolist())
        print(x.head(2).to_string(index=False))
        for c in x.columns:
            if "nll" in c.lower():
                print(c, pd.to_numeric(x[c], errors="coerce").mean())

print("\n=== RETAINED FUNCTIONS ===")
for k, v in list(globals().items()):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("\n=== RELEVANT RETAINED VARIABLES ===")
for k, v in list(globals().items()):
    if any(s in k.lower() for s in
           ("candidate", "target", "impulse", "causal", "direction",
            "recipe", "calib", "coef", "decoder", "feature", "channel")):
        if inspect.isfunction(v):
            continue
        desc = type(v).__name__
        if hasattr(v, "shape"):
            desc += " " + str(v.shape)
        elif isinstance(v, dict):
            desc += " keys=" + str(list(v.keys())[:35])
        elif isinstance(v, (list, tuple)):
            desc += " length=" + str(len(v))
        print(k, desc)

notes = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK EVIDENCE ===")
print(notes[:14000])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 140: audit after reversed-channel rollout\n"
            "- Forward 9: insulin x2 -2.750, half +1.370; NE x2 +0.210, half +0.095.\n"
            "- Reversed channels modestly strengthen insulin but fail NE; no further edits in this run.\n"
            "- Recovering architecture, official factual constraint, and retained calibration machinery.\n"
            "- Python runs: 140; forward passes: 9.\n")
```