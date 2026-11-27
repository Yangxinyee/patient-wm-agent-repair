The latest rollout again shows larger dose responses after the edit: insulin −10.321/+5.436 and norepinephrine +1.320/−1.543. These are predictions, not proof of the true effects. I’ll recover the record-based estimates and validation groups before deciding how to spend the remaining passes. Agreement across independent groups would support keeping this edit; inconsistent effects or inadequate factual safety would argue against it.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD-BASED ESTIMATES ===")
for final_name290 in ["kernel_targets.csv", "dml_results.csv",
                      "dynamic_iv.csv", "proxy_iv.csv", "sensor_iv.csv"]:
    final_path290 = Path("/work") / final_name290
    if final_path290.exists():
        final_df290 = pd.read_csv(final_path290)
        print(final_name290, "shape:", final_df290.shape)
        print(final_df290.head(8).to_string(index=False))

print("\n=== FACTUAL AUDIT ===")
final_audit290 = pd.read_csv("/work/final_factual_audit289.csv")
print("shape:", final_audit290.shape)
print(final_audit290.head(10).to_string(index=False))

print("\n=== RECENT FORWARD FILES AND STAY GROUPS ===")
for final_dir290 in ["fwd_40", "fwd_42", "fwd_43",
                     "fwd_44", "fwd_45", "fwd_46"]:
    final_p290 = Path("/work") / final_dir290
    if not final_p290.exists():
        continue
    print(final_dir290, sorted(x.name for x in final_p290.iterdir()))
    for final_csv290 in sorted(final_p290.glob("*.csv")):
        final_d290 = pd.read_csv(final_csv290)
        final_idcols290 = [
            c for c in final_d290.columns
            if c.lower() in ("stay", "stay_id", "stayid")
        ]
        print(" ", final_csv290.name, final_d290.shape,
              "columns:", list(final_d290.columns))
        if final_idcols290:
            final_ids290 = sorted(final_d290[final_idcols290[0]].unique().tolist())
            print("  stays:", len(final_ids290),
                  "first:", final_ids290[:3], "last:", final_ids290[-3:])

final_globals290 = dict(globals())
print("\n=== AVAILABLE RECOVERY HELPERS ===")
print(sorted(
    k for k, v in final_globals290.items()
    if callable(v) and not k.startswith("final_")
    and any(t in k.lower() for t in
            ("candidate", "calib", "restore", "kernel", "safety", "head"))
))
print("\nCurrent edited tensors:", edited())

with open("/work/notes.md", "a") as final_notes290:
    final_notes290.write(
        "\n## Run 290: final evidence recovery\n"
        "- Forward 44 / fwd_46, current hash e9fb2db92f: insulin "
        "-10.321/+5.436; NE +1.320/-1.543 on 100 tuning stays.\n"
        "- These are model predictions, not intervention truth. "
        "Recovering record-derived estimates and validation groups.\n"
        "- Current factual pooled dNLL +0.00334197; prior stay-bootstrap "
        "95% interval [0.00189249, 0.00475604]. No weights changed.\n"
        "- Counts: 290 Python runs, 44 forward passes; 10 and 6 remain.\n"
    )
```