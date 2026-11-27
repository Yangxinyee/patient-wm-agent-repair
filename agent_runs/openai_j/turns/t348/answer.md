This batch again shows a stronger insulin response after editing, while the norepinephrine estimates vary across stays. These are model predictions, not evidence of the true effects. I’ll consolidate the validation groups and recover the records-based estimates without changing weights. Agreement across groups would support keeping this candidate; a factual-fit violation or mismatch with those estimates would argue against it.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import re

root = Path("/work")
arch = Path("/model/ARCHITECTURE.md").read_text()
schema = Path("/data/README.md").read_text()
print("ARCHITECTURE (re-read):")
print(arch[:14000])
print("\nRECORD SCHEMA:")
print(schema[:5000])

print("\nCURRENT EDITS:")
def as_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

current_names = list(edited())
print(current_names)
backup = np.load(root / "safety_candidate285.npz", allow_pickle=False)
print("Safety archive keys:", backup.files)
for name in current_names:
    a = as_numpy(load(name))
    if name in backup.files:
        print(name, "shape", a.shape,
              "max difference from safety archive",
              float(np.max(np.abs(a - backup[name]))))
    else:
        print(name, "not directly keyed in safety archive")

print("\nCOMPACT RECORD-BASED ESTIMATES:")
for filename in ["kernel_targets.csv", "dynamic_iv.csv", "dml_results.csv",
                 "sensor_iv.csv", "proxy_iv.csv", "final_factual_audit289.csv"]:
    p = root / filename
    if p.exists():
        df = pd.read_csv(p)
        print("\n", filename, "shape", df.shape, "columns", list(df.columns))
        print(df.head(6).to_string(index=False)[:2300])

print("\nLATEST FORWARD FILES AND STAY GROUPS:")
for k in range(39, 49):
    folder = root / f"fwd_{k}"
    if not folder.exists():
        continue
    print("\n", folder.name, sorted(p.name for p in folder.iterdir()))
    for filename in ["dose_edit_edited.csv", "factual_edited.csv"]:
        p = folder / filename
        if not p.exists():
            continue
        df = pd.read_csv(p)
        id_columns = [c for c in df.columns
                      if c.lower() in ("stay", "stay_id", "stayid")]
        if id_columns:
            ids = sorted(df[id_columns[0]].dropna().unique().tolist())
            print(filename, "rows", len(df), "stays", len(ids),
                  "first/last", ids[:3], ids[-3:])
        if k == 48:
            print("columns:", list(df.columns))
            print(df.head(4).to_string(index=False)[:1600])

notes = (root / "notes.md").read_text()
hits = [m for m in re.finditer(
    r"(?im)^.*(?:identif|causal estimate|kernel target|instrument|true effect).*$",
    notes)]
print("\nSELECTED RECENT EVIDENCE NOTES:")
for m in hits[-8:]:
    print(notes[max(0, m.start()-120):min(len(notes), m.end()+450)])

with (root / "notes.md").open("a") as f:
    f.write(
        "\n## Run 293: final evidence consolidation\n"
        "- Forward 46 / fwd_48, current hash e9fb2db92f: insulin "
        "-9.604/+4.891; NE +1.721/-0.822 on 100 tuning stays.\n"
        "- These are model predictions, not observed intervention effects.\n"
        "- Re-read architecture/schema and inspected estimate files, "
        "validation groups, and current candidate identity; no weights changed.\n"
        "- Counts: 293 Python runs, 46 forward passes; 7 and 4 remain.\n"
    )
```