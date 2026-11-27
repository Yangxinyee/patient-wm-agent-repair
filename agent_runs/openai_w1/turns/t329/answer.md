The recovered notes show stronger dose associations after adjusting for treatment assignment, but those associations alone do not establish the intervention effects. The output was cut off before the current edit’s validation. I’ll retrieve that evidence and the new projection objects compactly; this should establish a reliable baseline before any further weight change.

```python
from pathlib import Path
import re, numpy as np, pandas as pd, torch

notes288 = Path("/work/notes.md").read_text()
print("Run 288/300; forwards 30/50; no weight changes.")
print("=== CURRENT SAFE EDIT EVIDENCE ===")
for num in [234, 235, 236, 237, 238]:
    m = re.search(r"(?m)^##[^\n]*\b"+str(num)+r"\b[^\n]*", notes288)
    if m:
        end = re.search(r"(?m)^## ", notes288[m.end():])
        text = notes288[m.start():m.end()+end.start()] if end else notes288[m.start():]
        print(text[:1400])

print("=== LATER RECORD-EFFECT CONCLUSIONS ===")
lines288 = [s for s in notes288.splitlines()
            if re.search(r"(?i)(causal|true.effect|dynamic.*(effect|gain)|effect.*(dynamic|target)|record.*(gain|estimate)|impulse|decay.*effect|identified|steady.state)", s)]
for s in lines288[-18:]:
    print(s[:330])

print("=== AVAILABLE OBJECTS ===")
for name in sorted(list(globals())):
    if re.search(r"(236|257|266|270|285|286)$", name):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "dict", [(str(k), getattr(v, "shape", type(v).__name__))
                                 for k, v in list(obj.items())[:8]])
        elif isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif callable(obj):
            print(name, "function args", obj.__code__.co_varnames[:obj.__code__.co_argcount])
        else:
            print(name, type(obj).__name__, str(obj)[:180])

print("=== FORWARD 24–27 SUMMARIES ===")
for k in [24, 25, 26, 27]:
    for p in sorted(Path(f"/work/fwd_{k}").glob("*.csv")):
        d = pd.read_csv(p)
        if "factual" in p.name:
            cc = [c for c in d.columns if "nll" in c.lower()]
            print(k, p.name, len(d), d[cc].mean(numeric_only=True).to_dict())
        else:
            print(k, p.name, list(d.columns))
            print(d.head(1).to_string(index=False))

print("=== EDIT MAGNITUDES ===")
for name in edited():
    delta = np.asarray(load(name)) - np.asarray(load_original(name))
    print(name, tuple(delta.shape), "delta RMS", float(np.sqrt(np.mean(delta**2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 288\nRun287 output truncation prevented recovery of current validation. Compact retrieval of safe236 evidence, record-effect conclusions, corrected projections and forward summaries. No weights changed. Counts: 288 Python runs, 30 forwards.\n")
```
