The architecture confirms that the edits directly affect only value-bin predictions; rollout feedback can still change later states. Earlier regressions also show why treatment associations alone are insufficient. I’ll recover the stronger identification evidence and exact stay IDs before testing the new rollout.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("FORWARD-22 STAYS:", audit226_stays)

audit227_notes = Path("/work/notes.md").read_text()
audit227_start = audit227_notes.find("## Causal identification checks")
print("\nIDENTIFICATION EVIDENCE")
print(audit227_notes[audit227_start:audit227_start + 4500])

print("\nKERNEL TARGETS")
audit227_targets = pd.read_csv("/work/kernel_targets.csv")
print(audit227_targets.head(12).to_string(index=False))

print("\nRETAINED CONSTRUCTION OBJECTS")
audit227_names = sorted(
    n for n in globals()
    if n.startswith(("head177", "map192", "map224", "map225", "cal190"))
)
for audit227_name in audit227_names[:65]:
    audit227_obj = globals()[audit227_name]
    print(audit227_name, type(audit227_obj).__name__,
          getattr(audit227_obj, "shape", ""),
          list(audit227_obj.keys())[:6] if isinstance(audit227_obj, dict) else "")

print("\nCHECKPOINT KEYS")
for audit227_file in ("validated_forward21.npz", "map_gain8_candidate225.npz",
                      "best_insulin_v1.npz", "best_combined_v1.npz"):
    with np.load("/work/" + audit227_file) as audit227_ckpt:
        print(audit227_file, audit227_ckpt.files)

with open("/work/notes.md", "a") as audit227_log:
    audit227_log.write(
        "\n## Restart audit, runs 226–227\n"
        "- Re-read architecture and recovered checkpoint/evidence objects; no weights changed.\n"
        "- Forward 22 gain-8 candidate factual value dNLL +0.00432 on 100 tuning stays.\n"
        "- This is narrowly within tolerance; rollout improvement remains unverified.\n"
    )
```
