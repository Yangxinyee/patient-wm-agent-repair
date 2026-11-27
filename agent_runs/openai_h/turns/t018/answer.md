The architecture supports a targeted dose-magnitude edit that retains drug identity, presence, and timing. However, the output was truncated before the saved validation results. I’ll retrieve those concisely; exact zero rollout shifts and small factual-NLL changes would support keeping the edit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from pprint import pprint

for audit_k in (3, 4, 5):
    audit_dir = Path(f"/work/fwd_{audit_k}")
    print(f"\n=== FORWARD {audit_k} ===")
    for audit_path in sorted(audit_dir.glob("*.csv")):
        audit_df = pd.read_csv(audit_path)
        print(audit_path.name, audit_df.shape, list(audit_df.columns))
        if "dose_edit" in audit_path.name:
            print(audit_df.head(2).to_string(index=False))
            audit_group = [
                c for c in audit_df.columns
                if c not in {"stay", "stay_id", "hour", "event", "target"}
                and not pd.api.types.is_numeric_dtype(audit_df[c])
                and audit_df[c].nunique() <= 8
            ]
            audit_num = audit_df.select_dtypes(include=np.number).columns
            if audit_group:
                print(audit_df.groupby(audit_group, dropna=False)[audit_num]
                      .agg(["mean", "min", "max"]).to_string())
            else:
                print(audit_df[audit_num].agg(["mean", "min", "max"]).to_string())
        else:
            audit_nll = [c for c in audit_df.columns if "nll" in c.lower()]
            print("Mean NLL:", audit_df[audit_nll].mean().to_dict())

print("\n=== TUNING ID BATCHES ===")
audit_ids = sorted(tuning["stay_id"].unique().tolist())
for audit_start in range(0, len(audit_ids), 100):
    print(f"Batch {audit_start // 100 + 1}:", audit_ids[audit_start:audit_start + 100])

print("\n=== MAIN-SENSOR ADJUSTED EFFECT ESTIMATES ===")
for audit_label, audit_results in [
    ("Nonlinear held-out", dml_results),
    ("Stronger adjustment", strong_results),
]:
    print(audit_label)
    for audit_event in ("glucose_finger", "map_arterial"):
        print(audit_event)
        pprint(audit_results[audit_event], width=110, compact=True)

print("\n=== PANEL AND HISTORY DEFINITIONS ===")
print("Panel index:", panel.index.names, "columns:", list(panel.columns))
print("History columns:", list(Hrich.columns))

with Path("/work/notes.md").open("a") as audit_f:
    audit_f.write(
        "\n## Python run 12: focused validation recovery\n"
        "- Run 11 output truncated before saved forward results; recovering "
        "forward 3–5 tables and main-sensor adjusted estimates without reprinting architecture.\n"
        "- Forward passes remain 5; no additional weight edits.\n"
    )
```