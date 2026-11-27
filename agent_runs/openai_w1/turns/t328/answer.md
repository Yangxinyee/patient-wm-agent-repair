The last fit recovered some dose sensitivity, but stronger factual anchoring sharply reduced it. Local reading sensitivity alone does not establish rollout stability or factual safety. I’ll recover the record-based effect estimates and prior validation before choosing a small, reversible test. The hypothesis is supported only if dose responses improve while factual NLL remains within the 0.005-nat limit.

```python
from pathlib import Path
import re, numpy as np, pandas as pd, torch

print("RECOVERY: Python run 287/300; forward passes 30/50; no weight changes.")
skill_path = Path("/Users/<user>/.codex/skills/evaluate-engineering-decisions/SKILL.md")
if skill_path.exists():
    print("=== ENGINEERING DECISION SKILL ===")
    print(skill_path.read_text()[:16000])

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes287 = Path("/work/notes.md").read_text()
print("=== EARLY RECORD IDENTIFICATION ===")
print(notes287[:10500])
for pattern in [r"(?m)^##[^\n]*\b236\b", r"(?m)^##[^\n]*\b257\b", r"(?m)^##[^\n]*\b286\b"]:
    match = re.search(pattern, notes287)
    if match:
        print("\n=== NOTEBOOK SECTION:", match.group(0), "===")
        print(notes287[match.start():match.start()+4200])

print("=== COMPACT RECOVERY OBJECTS ===")
for name in sorted(list(globals())):
    if re.search(r"(236|257|266|270|285|286)$", name):
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "dict", [(str(k), getattr(v, "shape", type(v).__name__))
                                 for k, v in list(obj.items())[:18]])
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, pd.DataFrame):
            print(name, "DataFrame", obj.shape, list(obj.columns))
        elif callable(obj):
            print(name, "function", obj.__code__.co_varnames[:obj.__code__.co_argcount])
        else:
            print(name, type(obj).__name__, str(obj)[:450])

print("=== RECENT FORWARD FILE SCHEMAS/SUMMARIES ===")
for k in [24, 25, 26, 27, 28, 29]:
    folder = Path(f"/work/fwd_{k}")
    for p in sorted(folder.glob("*.csv")):
        df287 = pd.read_csv(p)
        print(k, p.name, df287.shape, list(df287.columns))
        if "factual" in p.name:
            nll_cols287 = [c for c in df287.columns if "nll" in c.lower()]
            print("NLL means:", df287[nll_cols287].mean(numeric_only=True).to_dict())
        else:
            print(df287.head(5).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 287\nRestart recovery of architecture, record-effect identification, safe236 evidence and corrected projection objects. No weights changed. Counts: 287 Python runs, 30 forwards.\n")
```
