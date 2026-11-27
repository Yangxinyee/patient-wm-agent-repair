The insulin-only baseline is restored. The earlier analysis found substantial insulin effects, but I need the final record-based targets and baseline validation results before choosing another edit. I’ll recover those summaries compactly; they will show which remaining mismatch is worth addressing.

```python
import re
import pandas as pd

sections105 = re.split(r"\n## ", Path("/work/notes.md").read_text())
wanted105 = []
for sec105 in sections105:
    heading105 = sec105.splitlines()[0]
    match105 = re.search(r"Run\s+(\d+)", heading105)
    run105 = int(match105.group(1)) if match105 else -1
    if run105 in (13, 14, 15, 16, 17, 18, 19, 20, 58, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80):
        wanted105.append(sec105[:650])
print("SELECTED EVIDENCE AND VALIDATION NOTES")
print("\n\n".join(wanted105)[-5900:])

snapshot105 = list(globals().items())
print("FUNCTIONS", [k for k, v in snapshot105 if callable(v) and not k.startswith("_")])
print("TARGET/EDIT OBJECTS")
for k105, v105 in snapshot105:
    if k105.endswith("105") or k105.startswith("_"):
        continue
    if (k105.endswith(("77","82","95","100","102","103"))
        or any(s in k105.lower() for s in ("target", "causal", "baseline", "beta", "rho"))):
        if hasattr(v105, "shape"):
            print(k105, type(v105).__name__, tuple(v105.shape))
        elif isinstance(v105, dict):
            print(k105, "dict keys", list(v105)[:12])
        elif isinstance(v105, (int, float, str)):
            print(k105, str(v105)[:130])
print("FORWARD FILE STRUCTURE")
for file105 in ("/work/fwd_9/factual_edited.csv",
                "/work/fwd_10/dose_edit_edited.csv",
                "/work/fwd_13/factual_original.csv"):
    if Path(file105).exists():
        frame105 = pd.read_csv(file105)
        print(file105, frame105.shape, frame105.columns.tolist())
        print(frame105.head(2).to_string(index=False))
print("BASELINE DELTA", float(np.linalg.norm(archive104["delta"])))
with Path("/work/notes.md").open("a") as f105:
    f105.write("\n## Run 105: baseline restored; recover decision evidence\n"
               "Restored baseline77 delta after trial103 factual failure. "
               "Recovering final causal targets, validated baseline and retained helper interfaces compactly. "
               "Python runs: 105; forwards: 13.\n")
```