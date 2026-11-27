The latest cohort’s NLL increased by 0.00623, above the 0.005 limit. The current edit is therefore not safe to submit. I’ll recover the causal estimates, saved candidates, and cohort results before reducing the edit. A smaller edit would be acceptable only if it retains useful dose responses and restores a factual-fit margin.

```python
from pathlib import Path
import inspect, re, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK HEADINGS ===")
for line in notes.splitlines():
    if line.startswith("#"):
        print(line)

print("\n=== EARLY CAUSAL EVIDENCE ===")
sections = re.split(r"(?m)(?=^## )", notes)
selected = [s for s in sections
            if any(t in s[:500].lower() for t in
                   ["causal", "identif", "instrument", "effect estimate", "target", "safe fallback"])]
text = "\n".join(selected)
print(text[:16000] if text else notes[:12000])

print("\n=== RECENT FACTUAL RESULTS ===")
for k in range(30, 39):
    folder = Path(f"/work/fwd_{k}")
    a, b = folder/"factual_original.csv", folder/"factual_edited.csv"
    if a.exists() and b.exists():
        x, y = pd.read_csv(a), pd.read_csv(b)
        nc = [c for c in x.columns if "nll" in c.lower()]
        print(k, "columns", list(x.columns))
        for c in nc:
            if c in y:
                print(c, "original", float(x[c].mean()),
                      "edited", float(y[c].mean()),
                      "delta", float(y[c].mean()-x[c].mean()))
        for c in ["stay", "stay_id"]:
            if c in x:
                print("stays", sorted(x[c].unique().tolist())[:5],
                      "...", sorted(x[c].unique().tolist())[-5:])

print("\n=== LIVE INTERFACES ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"notes", "sections", "selected", "text"}:
        continue
    if inspect.isfunction(obj):
        try:
            print("FUNCTION", name, inspect.signature(obj))
        except Exception:
            pass
    elif any(t in name.lower() for t in
             ["cache", "safe", "cand", "target", "r244", "r25", "checkpoint", "lastbin"]):
        if isinstance(obj, dict):
            print("DICT", name, "keys", list(obj.keys())[:25])
        elif isinstance(obj, (np.ndarray, pd.DataFrame)):
            print(type(obj).__name__, name, obj.shape)
        else:
            print(type(obj).__name__, name, str(obj)[:180])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 259: factual-budget recovery\n"
            "Forward38 current five-tensor checkpoint 8898bdc457 increased factual NLL "
            "by about .00623 on its 100-stay cohort, exceeding the .005 limit. "
            "No weights changed. Recovering causal targets, saved candidates, "
            "and pooled validation results before reducing the edit.\n")
```
